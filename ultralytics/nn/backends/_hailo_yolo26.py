# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Host-side YOLO26 decode for HailoBackend.

Lives next to ``hailo.py`` rather than inside it because the YOLO26 path is a NumPy reimplementation of
``Detect.postprocess`` (anchors + dist2bbox + sigmoid + topk) — the chip can't run topk/gather. Keeping
it separate keeps ``HailoBackend`` itself thin (loader + dispatch).
"""

from __future__ import annotations

import numpy as np

from ultralytics.utils import LOGGER


def classify_buffers(
    head_outputs: dict,
    output_names: list,
    out_bufs: list[np.ndarray],
    nc: int,
    strides: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pre-classify the 6 YOLO26 output buffers into per-stride box and class lists.

    Prefers the ``head_outputs`` metadata persisted at compile time — names + spatials in stride order —
    to avoid the channel-count ambiguity that breaks for ``nc == 4`` custom datasets (where every leaf
    has 4 channels). Falls back to channel-count + spatial-dim sort if the metadata is missing or names
    cannot be resolved (older HEFs / SDK rename).

    Args:
        head_outputs (dict): ``{"box_layers": [{"name", "spatial"}, ...], "cls_layers": [...]}`` in
            stride order, or ``{}`` to force the channel-count fallback.
        output_names (list): Output names from ``infer_model.outputs[i].name`` (runtime).
        out_bufs (list[np.ndarray]): Output buffers in the same order as ``output_names``.
        nc (int): Number of classes (used by the channel-count fallback).
        strides (list[int]): Per-level strides; length is the expected per-role count.

    Returns:
        (tuple[list[np.ndarray], list[np.ndarray]]): ``(box_bufs, cls_bufs)``, each of length
        ``len(strides)``, ordered by stride (largest spatial → smallest).

    Raises:
        RuntimeError: If neither name dispatch nor the channel-count fallback can resolve all
            ``2 * len(strides)`` buffers unambiguously.
    """
    n = len(strides)

    def _resolve_by_name(entries: list) -> list[np.ndarray] | None:
        """Return buffers in entry order, or None if any entry can't be matched."""
        name_to_buf = dict(zip(output_names, out_bufs))
        picked: list[np.ndarray] = []
        for entry in entries:
            want = entry.get("name")
            if want and want in name_to_buf:
                picked.append(name_to_buf[want])
                continue
            # Substring match — runtime SDK may prefix the metadata name (e.g. "yolo26m/conv61" ->
            # "conv61"). Only match in the prefix-tolerance direction (want is substring of nm); the
            # reverse would let a short runtime name collide with every longer metadata entry.
            hits = [b for nm, b in name_to_buf.items() if want and want in nm]
            if len(hits) == 1:
                picked.append(hits[0])
                continue
            return None
        return picked

    if head_outputs and len(head_outputs.get("box_layers", [])) == n and len(head_outputs.get("cls_layers", [])) == n:
        box = _resolve_by_name(head_outputs["box_layers"])
        cls = _resolve_by_name(head_outputs["cls_layers"])
        if box is not None and cls is not None:
            return box, cls

    # Fallback: channel-count + spatial sort. Ambiguous when nc == 4; warn the user.
    bufs_4d = [b if b.ndim == 4 else b[None, ...] for b in out_bufs]
    if nc == 4:
        LOGGER.warning(
            "Hailo: YOLO26 head_outputs metadata missing/unresolved with nc==4 — channel counts alone "
            "cannot disambiguate box vs cls heads. Re-export with the current Ultralytics version to "
            "embed head_outputs."
        )
    box_outs = [b for b in bufs_4d if b.shape[-1] == 4]
    cls_outs = [b for b in bufs_4d if b.shape[-1] == nc]
    if len(box_outs) != n or len(cls_outs) != n:
        raise RuntimeError(
            f"YOLO26 fallback dispatch expects {n} box (4 ch) + {n} cls ({nc} ch) heads, got "
            f"{len(box_outs)} / {len(cls_outs)}. Check metadata.yaml's nc, or re-export to embed "
            f"head_outputs."
        )
    box_outs.sort(key=lambda b: -b.shape[1])
    cls_outs.sort(key=lambda b: -b.shape[1])
    return box_outs, cls_outs


def postprocess(
    box_bufs: list[np.ndarray],
    cls_bufs: list[np.ndarray],
    strides: list[int],
    nc: int,
    max_det: int,
    conf: float,
) -> np.ndarray:
    """Host-side YOLO26 postprocess: anchors + dist2bbox + sigmoid + top-k.

    Mirrors ``Detect._inference`` followed by ``Detect.postprocess`` (for end2end models). YOLO26 has
    ``reg_max=1`` so DFL is ``nn.Identity`` — box conv outputs are direct 4-channel l/t/r/b distances
    in stride units. The xyxy box decode happens in stride units, then we scale by stride to get pixels.
    ``reg_max`` is hardcoded to 1 here (validated in ``HailoBackend.load_model``); a real DFL would need
    a ``softmax(reg_max) @ arange(reg_max)`` step before dist2bbox.

    Args:
        box_bufs (list[np.ndarray]): Per-stride box-reg conv outputs (NHWC, 4 channels), in stride
            order matching ``strides``. Pre-classified in ``HailoBackend.load_model`` from
            ``metadata['head_outputs']``.
        cls_bufs (list[np.ndarray]): Per-stride class conv outputs (NHWC, ``nc`` channels), in stride
            order matching ``strides``.
        strides (list[int]): Per-level strides, e.g. ``[8, 16, 32]``. Index 0 maps to the largest
            spatial feature map.
        nc (int): Number of classes.
        max_det (int): Top-k limit on (anchor, class) pairs.
        conf (float): Score threshold; predictions below this are dropped after top-k.

    Returns:
        (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]``, descending score.
    """
    # HailoRT emits (H, W, C) for batch=1 outputs. Add a leading batch dim if missing. The whole
    # postprocess (anchor sharing, conf filter, predictor's (1, N, 6) contract) is wired for B=1 — if a
    # multi-batch HEF ever reaches us, fail loudly rather than produce wrong results below.
    box_bufs = [b if b.ndim == 4 else b[None, ...] for b in box_bufs]
    cls_bufs = [c if c.ndim == 4 else c[None, ...] for c in cls_bufs]
    for buf in (*box_bufs, *cls_bufs):
        if buf.shape[0] != 1:
            raise NotImplementedError(
                f"YOLO26 host postprocess only supports batch=1, got buffer shape {buf.shape}. "
                f"Multi-batch decode is not implemented."
            )
    if len(box_bufs) != len(strides) or len(cls_bufs) != len(strides):
        raise RuntimeError(
            f"YOLO26 host postprocess expected {len(strides)} box / {len(strides)} cls buffers, "
            f"got {len(box_bufs)} / {len(cls_bufs)}."
        )

    all_boxes, all_scores = [], []
    for stride, box, cls in zip(strides, box_bufs, cls_bufs):
        B, H, W, _ = box.shape
        # Anchor centers in stride units: (x + 0.5, y + 0.5).
        ys = np.arange(H, dtype=np.float32) + 0.5
        xs = np.arange(W, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        ax = gx.reshape(-1)  # (H*W,)
        ay = gy.reshape(-1)
        dist = box.reshape(B, -1, 4)
        l = dist[..., 0]
        t = dist[..., 1]
        r = dist[..., 2]
        b_ = dist[..., 3]
        x1 = (ax - l) * stride
        y1 = (ay - t) * stride
        x2 = (ax + r) * stride
        y2 = (ay + b_) * stride
        boxes_stride = np.stack([x1, y1, x2, y2], axis=-1)  # (B, H*W, 4)
        scores_stride = 1.0 / (1.0 + np.exp(-cls.reshape(B, -1, nc)))  # sigmoid
        all_boxes.append(boxes_stride)
        all_scores.append(scores_stride)

    boxes = np.concatenate(all_boxes, axis=1)  # (B, N_total, 4)
    scores = np.concatenate(all_scores, axis=1)  # (B, N_total, nc)

    # End2end top-k over (anchor × class) pairs, mirrors Detect.postprocess + get_topk_index.
    B, N, _ = boxes.shape
    flat_scores = scores.reshape(B, -1)  # (B, N*nc)
    k = min(max_det, flat_scores.shape[1])

    if k < flat_scores.shape[1]:
        # argpartition with a negative kth selects the top-k in unsorted order without the
        # full-array negation that `-flat_scores` would copy.
        part = np.argpartition(flat_scores, -k, axis=1)[:, -k:]
        unsorted = np.take_along_axis(flat_scores, part, axis=1)
        order = np.argsort(-unsorted, axis=1)
        topk_idx = np.take_along_axis(part, order, axis=1)
    else:
        topk_idx = np.argsort(-flat_scores, axis=1)

    topk_scores = np.take_along_axis(flat_scores, topk_idx, axis=1)  # (B, k)
    anchor_idx = topk_idx // nc  # (B, k)
    cls_idx = (topk_idx % nc).astype(np.float32)  # (B, k)
    batch_arange = np.arange(B)[:, None]
    selected_boxes = boxes[batch_arange, anchor_idx]  # (B, k, 4)

    # Drop predictions below conf threshold. B is guaranteed to be 1 (asserted above), so we can filter
    # unconditionally and produce the (1, k_kept, 6) the predictor expects.
    out = np.concatenate([selected_boxes, topk_scores[..., None], cls_idx[..., None]], axis=-1)
    if conf > 0:
        mask = out[0, :, 4] > conf
        out = out[:, mask]
    return out.astype(np.float32, copy=False)
