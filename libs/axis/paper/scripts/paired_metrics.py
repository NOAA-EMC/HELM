"""Accuracy on an explicitly shared eligible destination population."""
import numpy as np

def common_cell_metrics(axis, esmc, truth, area, covered, mask, conservative):
    arrays = [np.asarray(a) for a in (axis['dst'], esmc['dst'], truth, area, covered, mask,
                                     axis['dst_frac'], esmc['dst_frac'])]
    if not arrays or any(a.shape != arrays[0].shape or a.ndim != 1 for a in arrays):
        raise ValueError('Paired fields have incompatible extents')
    av, ev, truth, area, covered, mask, af, ef = arrays
    eligible = (mask != 0) & (covered > 0) & (af > 0) & (ef > 0)
    if not conservative:
        eligible &= covered >= area * (1 - 1e-10)
    candidate = eligible.copy()
    eligible &= np.isfinite(av) & np.isfinite(ev) & np.isfinite(truth)
    result = {'eligible': int(eligible.sum()), 'excluded_nonfinite': int((candidate & ~eligible).sum()),
              'objective': 'cell_mean_transfer', 'population': 'intersection of both mapped wet populations'}
    if not eligible.any():
        result.update(status='unavailable', cause='No shared eligible cells')
        return result
    weights = covered[eligible]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('Invalid reference areas')
    def metrics(values):
        e = values[eligible] - truth[eligible]
        return {'rms_error': float(np.sqrt(np.average(e*e, weights=weights))),
                'mean_absolute_error': float(np.average(np.abs(e), weights=weights)),
                'bias': float(np.average(e, weights=weights)), 'max_error': float(np.max(np.abs(e)))}
    result.update(status='complete', axis=metrics(av), esmc=metrics(ev))
    return result
