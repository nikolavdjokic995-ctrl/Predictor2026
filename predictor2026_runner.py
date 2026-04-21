import csv
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - optional dependency in user QGIS env
    RandomForestClassifier = None


gdal.UseExceptions()

STANDARD_CLASSES = list(range(1, 8))
RF_RANDOM_STATE = 42
MAX_SAMPLES_PER_PERIOD = 12000
MIN_POSITIVE_SAMPLES = 25


def _open_array(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    return ds, arr, nodata


def _same_grid(ds, ref_ds, tol=1e-9):
    if ds.RasterXSize != ref_ds.RasterXSize or ds.RasterYSize != ref_ds.RasterYSize:
        return False
    gt = ds.GetGeoTransform()
    ref_gt = ref_ds.GetGeoTransform()
    if gt is None or ref_gt is None:
        return False
    if any(abs(float(a) - float(b)) > tol for a, b in zip(gt, ref_gt)):
        return False
    proj = (ds.GetProjection() or '').strip()
    ref_proj = (ref_ds.GetProjection() or '').strip()
    return proj == ref_proj


def _warp_to_reference(path, ref_ds, resample_alg=gdal.GRA_Bilinear):
    ref_gt = ref_ds.GetGeoTransform()
    minx = ref_gt[0]
    maxy = ref_gt[3]
    maxx = minx + ref_gt[1] * ref_ds.RasterXSize
    miny = maxy + ref_gt[5] * ref_ds.RasterYSize
    warped = gdal.Warp(
        '',
        str(path),
        format='MEM',
        width=ref_ds.RasterXSize,
        height=ref_ds.RasterYSize,
        outputBounds=(minx, miny, maxx, maxy),
        dstSRS=ref_ds.GetProjection(),
        resampleAlg=resample_alg,
    )
    if warped is None:
        raise RuntimeError(f"Could not align raster to reference grid: {path}")
    band = warped.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    return warped, arr, nodata


def _align_array(path, ref_ds, resample_alg=gdal.GRA_NearestNeighbour):
    ds, arr, nodata = _open_array(path)
    if not _same_grid(ds, ref_ds):
        ds, arr, nodata = _warp_to_reference(path, ref_ds, resample_alg=resample_alg)
    return ds, arr, nodata


def _reference_from_config(config):
    periods = config.get("lulc_periods", [])
    if not periods:
        raise RuntimeError("No LULC periods were found in the configuration.")
    ref_path = periods[0].get("start_layer_source") or periods[0].get("end_layer_source")
    if not ref_path:
        raise RuntimeError("Reference raster path is missing.")
    ds, arr, nodata = _open_array(ref_path)
    return ds, arr, nodata, ref_path


def _boundary_mask(boundary_source, ref_ds):
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    mem = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(ref_ds.GetGeoTransform())
    mem.SetProjection(ref_ds.GetProjection())
    mem.GetRasterBand(1).Fill(0)

    vector_ds = ogr.Open(str(boundary_source))
    if vector_ds is None:
        raise RuntimeError(f"Could not open boundary source: {boundary_source}")
    layer = vector_ds.GetLayer()
    gdal.RasterizeLayer(mem, [1], layer, burn_values=[1])
    return mem.GetRasterBand(1).ReadAsArray().astype(bool)


def _valid_mask(arr, nodata, boundary_mask):
    mask = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        mask &= arr != nodata
    if boundary_mask is not None:
        mask &= boundary_mask
    return mask


def _normalize(arr, mask, invert=False):
    out = np.zeros(arr.shape, dtype=np.float32)
    if not np.any(mask):
        return out
    vals = arr[mask].astype(np.float32)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        out[mask] = 0.5
    else:
        out[mask] = (vals - vmin) / (vmax - vmin)
    if invert:
        out[mask] = 1.0 - out[mask]
    return out


def _model_settings(config):
    settings = dict(config.get("model_settings") or {})
    strength = str(settings.get("neighborhood_strength", "Medium") or "Medium").strip().lower()
    if strength not in ("low", "medium", "high"):
        strength = "medium"
    return {
        "use_neighborhood": bool(settings.get("use_neighborhood", True)),
        "neighborhood_strength": strength.title(),
        "strength_value": {"low": 0.20, "medium": 0.40, "high": 0.65}[strength],
    }


def _latest_observed_period(config):
    periods = config.get("lulc_periods", [])
    if not periods:
        raise RuntimeError("No periods available.")

    def _end_year(period):
        try:
            return int(period.get("end_year", 0))
        except Exception:
            return 0

    return sorted(periods, key=_end_year)[-1]


def _validation_period(config):
    periods = [p for p in config.get("lulc_periods", []) if p.get("purpose") == "VALIDATION"]
    if not periods:
        return None

    def _end_year(period):
        try:
            return int(period.get("end_year", 0))
        except Exception:
            return 0

    return sorted(periods, key=_end_year)[-1]


def _training_rates(config, boundary_mask, ref_ds):
    transitions = config.get("transitions", [])
    exposures = {t["code"]: 0.0 for t in transitions}
    changes = {t["code"]: 0.0 for t in transitions}
    train_count = 0
    for period in config.get("lulc_periods", []):
        if period.get("purpose") != "TRAIN":
            continue
        train_count += 1
        _, a0, n0 = _align_array(period["start_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
        _, a1, n1 = _align_array(period["end_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
        try:
            years = max(1, int(period["end_year"]) - int(period["start_year"]))
        except Exception:
            years = 1
        valid = _valid_mask(a0, n0, boundary_mask)
        if n1 is not None:
            valid &= a1 != n1
        for transition in transitions:
            code = transition["code"]
            from_cls = int(transition["from"])
            to_cls = int(transition["to"])
            eligible = valid & (a0 == from_cls)
            exposures[code] += float(np.count_nonzero(eligible)) * float(years)
            changes[code] += float(np.count_nonzero(eligible & (a1 == to_cls)))
    rates = {}
    for transition in transitions:
        code = transition["code"]
        rates[code] = 0.0 if exposures[code] <= 0 else changes[code] / exposures[code]
    return rates, train_count


def _transition_score(suitability, neighborhood, to_cls, strength_value, use_neighborhood):
    if not use_neighborhood:
        return suitability
    class_boost = 1.35 if int(to_cls) == 1 else 0.85 if int(to_cls) in (4, 5) else 0.70
    neighborhood_component = np.clip(neighborhood * class_boost, 0.0, 1.0)
    suitability_weight = 1.0 - min(0.55, strength_value)
    neighborhood_weight = min(0.55, strength_value)
    return (suitability_weight * suitability) + (neighborhood_weight * neighborhood_component)


def _neighborhood_fraction(arr, target_class, valid_mask):
    binary = (arr == int(target_class)).astype(np.float32)
    padded = np.pad(binary, 1, mode="edge")
    total = np.zeros_like(binary, dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += padded[1 + dr:1 + dr + binary.shape[0], 1 + dc:1 + dc + binary.shape[1]]
    frac = total / 8.0
    frac[~valid_mask] = 0.0
    return frac


def _predictor_sources_for_year(config, target_year=None):
    layers = []
    for predictor in config.get("predictors", {}).get("static", []):
        path = predictor.get("layer_source")
        if path:
            layers.append((predictor.get("name", "static"), path, predictor.get("type", "static")))

    dynamic = config.get("predictors", {}).get("dynamic", [])
    grouped = {}
    for predictor in dynamic:
        name = predictor.get("name", "").strip()
        path = predictor.get("layer_source")
        if not name or not path:
            continue
        grouped.setdefault(name, []).append(predictor)

    for name, items in grouped.items():
        numeric = [item for item in items if str(item.get("year", "")).isdigit()]
        chosen = None
        if numeric and target_year is not None:
            chosen = min(numeric, key=lambda item: abs(int(item["year"]) - int(target_year)))
        elif numeric:
            chosen = sorted(numeric, key=lambda item: int(item["year"]))[-1]
        elif items:
            chosen = items[-1]
        if chosen:
            layers.append((name, chosen["layer_source"], chosen.get("type", "dynamic")))
    return layers


def _feature_stack_for_year(config, ref_ds, boundary_mask, target_year=None):
    layers = _predictor_sources_for_year(config, target_year=target_year)
    feature_names = []
    arrays = []
    for name, path, _ptype in layers:
        ds, arr, nodata = _align_array(path, ref_ds, resample_alg=gdal.GRA_Bilinear)
        mask = _valid_mask(arr, nodata, boundary_mask)
        invert = "distance" in name.lower()
        arrays.append(_normalize(arr, mask, invert=invert))
        feature_names.append(name)
    if not arrays:
        fallback = np.zeros((ref_ds.RasterYSize, ref_ds.RasterXSize, 1), dtype=np.float32)
        fallback[boundary_mask, 0] = 0.5
        return ["fallback_suitability"], fallback
    stack = np.stack(arrays, axis=-1).astype(np.float32)
    stack[~boundary_mask] = 0.0
    return feature_names, stack


def _mean_suitability_from_stack(feature_stack, boundary_mask):
    suitability = feature_stack.mean(axis=-1).astype(np.float32)
    suitability[~boundary_mask] = 0.0
    return suitability


def _fit_rf_models(config, ref_ds, boundary_mask, transitions):
    notes = []
    if RandomForestClassifier is None:
        notes.append("scikit-learn is not available in this QGIS Python environment; using mean predictor suitability fallback.")
        return {}, {}, notes

    models = {}
    feature_importance = {}
    train_periods = [p for p in config.get("lulc_periods", []) if p.get("purpose") == "TRAIN"]
    for transition in transitions:
        code = transition["code"]
        from_cls = int(transition["from"])
        to_cls = int(transition["to"])
        xs = []
        ys = []
        feature_names_ref = None
        for period in train_periods:
            try:
                period_year = int(period.get("start_year"))
            except Exception:
                period_year = None
            feature_names, feature_stack = _feature_stack_for_year(config, ref_ds, boundary_mask, target_year=period_year)
            if feature_names_ref is None:
                feature_names_ref = feature_names
            _, start_arr, start_nodata = _align_array(period["start_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
            _, end_arr, end_nodata = _align_array(period["end_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
            valid = _valid_mask(start_arr, start_nodata, boundary_mask)
            if end_nodata is not None:
                valid &= end_arr != end_nodata
            eligible = valid & (start_arr == from_cls)
            if not np.any(eligible):
                continue
            pos_idx = np.flatnonzero(eligible & (end_arr == to_cls))
            neg_idx = np.flatnonzero(eligible & (end_arr != to_cls))
            if pos_idx.size == 0 or neg_idx.size == 0:
                continue
            rng = np.random.default_rng(RF_RANDOM_STATE + abs(hash(code)) % 997)
            max_pos = min(pos_idx.size, MAX_SAMPLES_PER_PERIOD // 2)
            max_neg = min(neg_idx.size, max_pos * 2, MAX_SAMPLES_PER_PERIOD - max_pos)
            pos_sel = rng.choice(pos_idx, size=max_pos, replace=False) if pos_idx.size > max_pos else pos_idx
            neg_sel = rng.choice(neg_idx, size=max_neg, replace=False) if neg_idx.size > max_neg else neg_idx
            sample_idx = np.concatenate([pos_sel, neg_sel])
            sample_x = feature_stack.reshape(-1, feature_stack.shape[-1])[sample_idx]
            sample_y = np.concatenate([
                np.ones(pos_sel.shape[0], dtype=np.uint8),
                np.zeros(neg_sel.shape[0], dtype=np.uint8),
            ])
            xs.append(sample_x)
            ys.append(sample_y)
        if not xs:
            notes.append(f"Transition {code}: insufficient training samples for RF; using fallback suitability.")
            continue
        x_all = np.vstack(xs)
        y_all = np.concatenate(ys)
        positives = int(np.count_nonzero(y_all == 1))
        negatives = int(np.count_nonzero(y_all == 0))
        if positives < MIN_POSITIVE_SAMPLES or negatives < MIN_POSITIVE_SAMPLES:
            notes.append(f"Transition {code}: only {positives} positive / {negatives} negative samples; using fallback suitability.")
            continue
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RF_RANDOM_STATE,
            class_weight="balanced_subsample",
        )
        model.fit(x_all, y_all)
        models[code] = {
            "model": model,
            "feature_names": feature_names_ref or [f"feature_{i + 1}" for i in range(x_all.shape[1])],
            "sample_count": int(x_all.shape[0]),
            "positive_count": positives,
            "negative_count": negatives,
        }
        if hasattr(model, "feature_importances_"):
            feature_importance[code] = {
                name: float(value) for name, value in zip(models[code]["feature_names"], model.feature_importances_.tolist())
            }
    if models:
        notes.append(f"RF suitability trained for {len(models)} transition(s).")
    return models, feature_importance, notes


def _suitability_maps_for_year(config, ref_ds, boundary_mask, rf_models, target_year=None):
    feature_names, feature_stack = _feature_stack_for_year(config, ref_ds, boundary_mask, target_year=target_year)
    fallback = _mean_suitability_from_stack(feature_stack, boundary_mask)
    flat = feature_stack.reshape(-1, feature_stack.shape[-1])
    suitability_maps = {}
    confidence_maps = {}
    for code, payload in rf_models.items():
        try:
            probs = payload["model"].predict_proba(flat)[:, 1].reshape(fallback.shape).astype(np.float32)
            probs[~boundary_mask] = 0.0
            suitability_maps[code] = probs
            confidence_maps[code] = np.abs(probs - 0.5) * 2.0
        except Exception:
            suitability_maps[code] = fallback
            confidence_maps[code] = np.zeros_like(fallback, dtype=np.float32)
    return {
        "feature_names": feature_names,
        "feature_stack": feature_stack,
        "fallback": fallback,
        "suitability_by_transition": suitability_maps,
        "confidence_by_transition": confidence_maps,
    }


def _apply_one_year(current, valid_mask, default_suitability, suitability_by_transition, transitions, annual_rates, multipliers, model_settings):
    updated = current.copy()
    changed_mask = np.zeros(current.shape, dtype=bool)
    year_confidence = np.zeros(current.shape, dtype=np.float32)

    grouped = {}
    for transition in transitions:
        grouped.setdefault(int(transition["from"]), []).append(transition)

    neighborhood_cache = {}
    for from_cls, items in grouped.items():
        for transition in items:
            code = transition["code"]
            to_cls = int(transition["to"])
            multiplier = float(multipliers.get(code, 1.0) or 1.0)
            rate = float(annual_rates.get(code, 0.0)) * multiplier
            if rate <= 0:
                continue
            candidates_mask = valid_mask & (updated == from_cls) & (~changed_mask)
            flat_candidates = np.where(candidates_mask.ravel())[0]
            if flat_candidates.size == 0:
                continue
            n_change = int(round(rate * flat_candidates.size))
            if n_change <= 0:
                continue
            if to_cls not in neighborhood_cache:
                neighborhood_cache[to_cls] = _neighborhood_fraction(updated, to_cls, valid_mask)
            neighborhood = neighborhood_cache[to_cls]
            suitability = suitability_by_transition.get(code, default_suitability)
            score = _transition_score(
                suitability,
                neighborhood,
                to_cls=to_cls,
                strength_value=model_settings.get("strength_value", 0.40),
                use_neighborhood=model_settings.get("use_neighborhood", True),
            )
            flat_scores = score.ravel()[flat_candidates]
            order = np.argsort(-flat_scores, kind='stable')
            chosen = flat_candidates[order[: min(n_change, flat_candidates.size)]]
            rows, cols = np.unravel_index(chosen, updated.shape)
            updated[rows, cols] = to_cls
            changed_mask[rows, cols] = True
            year_confidence[rows, cols] = np.clip(flat_scores[order[: min(n_change, flat_candidates.size)]], 0.0, 1.0)
    return updated, changed_mask, year_confidence


def _predict_to_year(start_array, start_year, target_year, valid_mask, default_suitability, suitability_by_transition, transitions, annual_rates, multipliers, model_settings):
    current = start_array.copy()
    changed_any = np.zeros(current.shape, dtype=np.uint8)
    confidence = np.zeros(current.shape, dtype=np.float32)
    for _year in range(int(start_year), int(target_year)):
        current, changed_mask, year_confidence = _apply_one_year(
            current,
            valid_mask,
            default_suitability,
            suitability_by_transition,
            transitions,
            annual_rates,
            multipliers,
            model_settings,
        )
        changed_any[changed_mask] = 1
        confidence = np.maximum(confidence, year_confidence)
    return current, changed_any, confidence


def _confusion_matrix(observed, predicted, valid_mask, classes=None):
    if classes is None:
        classes = STANDARD_CLASSES
    classes = [int(c) for c in classes]
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    observed = observed.astype(np.int32)
    predicted = predicted.astype(np.int32)
    use = valid_mask.copy()
    for row_idx, real_cls in enumerate(classes):
        row_obs = use & (observed == real_cls)
        if not np.any(row_obs):
            continue
        for col_idx, pred_cls in enumerate(classes):
            matrix[row_idx, col_idx] = int(np.count_nonzero(row_obs & (predicted == pred_cls)))
    total = int(matrix.sum())
    diag = int(np.trace(matrix))
    oa = float(diag) / float(total) if total > 0 else 0.0
    row_sums = matrix.sum(axis=1).astype(np.float64)
    col_sums = matrix.sum(axis=0).astype(np.float64)
    pe = float((row_sums * col_sums).sum()) / float(total * total) if total > 0 else 0.0
    kappa = 0.0
    if total > 0 and abs(1.0 - pe) > 1e-12:
        kappa = (oa - pe) / (1.0 - pe)
    precision_by_class = {}
    recall_by_class = {}
    for idx, cls in enumerate(classes):
        tp = float(matrix[idx, idx])
        precision = tp / float(col_sums[idx]) if col_sums[idx] > 0 else 0.0
        recall = tp / float(row_sums[idx]) if row_sums[idx] > 0 else 0.0
        precision_by_class[str(cls)] = precision
        recall_by_class[str(cls)] = recall
    macro_precision = float(np.mean(list(precision_by_class.values()))) if precision_by_class else 0.0
    macro_recall = float(np.mean(list(recall_by_class.values()))) if recall_by_class else 0.0
    return {
        "classes": classes,
        "matrix": matrix,
        "overall_accuracy": oa,
        "kappa": float(kappa),
        "total_pixels": total,
        "correct_pixels": diag,
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def _write_confusion_csv(path, stats):
    classes = stats["classes"]
    matrix = stats["matrix"]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(["observed\\predicted"] + classes + ["row_total", "recall"])
        for i, cls in enumerate(classes):
            writer.writerow([
                cls,
                *matrix[i].tolist(),
                int(matrix[i].sum()),
                f"{stats['recall_by_class'].get(str(cls), 0.0):.6f}",
            ])
        writer.writerow([
            "col_total",
            *matrix.sum(axis=0).astype(int).tolist(),
            int(matrix.sum()),
            "",
        ])
        writer.writerow([
            "precision",
            *[f"{stats['precision_by_class'].get(str(cls), 0.0):.6f}" for cls in classes],
            "",
            "",
        ])


def _write_raster(path, ref_ds, array, nodata=-9999, gdal_type=gdal.GDT_Int16):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal_type, options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(array)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def _write_feature_importance_json(path, feature_importance):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(feature_importance, handle, indent=2)


def _write_feature_importance_csv(path, feature_importance, rf_models):
    rows = []
    aggregated = {}
    total_weight = 0.0
    for code, importance_map in feature_importance.items():
        payload = rf_models.get(code, {})
        sample_count = int(payload.get("sample_count", 0))
        positive_count = int(payload.get("positive_count", 0))
        negative_count = int(payload.get("negative_count", 0))
        weight = float(sample_count if sample_count > 0 else 1)
        total_weight += weight
        total_importance = sum(float(v) for v in importance_map.values()) or 1.0
        for predictor, importance in sorted(importance_map.items(), key=lambda kv: kv[1], reverse=True):
            importance = float(importance)
            rows.append({
                "scope": "transition",
                "transition": code,
                "predictor": predictor,
                "importance": importance,
                "importance_normalized": importance / total_importance,
                "sample_count": sample_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
            })
            aggregated[predictor] = aggregated.get(predictor, 0.0) + importance * weight

    if aggregated:
        grand_total = sum(aggregated.values()) or 1.0
        for predictor, value in sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True):
            rows.append({
                "scope": "overall",
                "transition": "ALL",
                "predictor": predictor,
                "importance": value / (total_weight or 1.0),
                "importance_normalized": value / grand_total,
                "sample_count": int(total_weight),
                "positive_count": "",
                "negative_count": "",
            })

    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "scope", "transition", "predictor", "importance", "importance_normalized",
            "sample_count", "positive_count", "negative_count"
        ])
        writer.writeheader()
        writer.writerows(rows)


def _run_validation(config, output_dir, ref_ds, boundary_mask, annual_rates, model_settings, rf_models):
    validation_period = _validation_period(config)
    if validation_period is None:
        return None

    start_year = int(validation_period.get("start_year"))
    end_year = int(validation_period.get("end_year"))
    _, start_arr, start_nodata = _align_array(validation_period["start_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
    _, observed, observed_nodata = _align_array(validation_period["end_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
    valid_mask = _valid_mask(start_arr, start_nodata, boundary_mask)
    if observed_nodata is not None:
        valid_mask &= observed != observed_nodata

    transitions = config.get("transitions", [])
    suitability_payload = _suitability_maps_for_year(config, ref_ds, boundary_mask, rf_models, target_year=end_year)
    predicted, _changed, _confidence = _predict_to_year(
        start_arr,
        start_year,
        end_year,
        valid_mask,
        suitability_payload["fallback"],
        suitability_payload["suitability_by_transition"],
        transitions,
        annual_rates,
        {},
        model_settings,
    )
    stats = _confusion_matrix(observed, predicted, valid_mask, classes=STANDARD_CLASSES)
    csv_path = output_dir / "confusion_matrix_BAU.csv"
    _write_confusion_csv(csv_path, stats)
    result = {
        "scenario": "BAU",
        "validation_period": f"{start_year}->{end_year}",
        "overall_accuracy": stats["overall_accuracy"],
        "kappa": stats["kappa"],
        "macro_precision": stats["macro_precision"],
        "macro_recall": stats["macro_recall"],
        "precision_by_class": stats["precision_by_class"],
        "recall_by_class": stats["recall_by_class"],
        "total_pixels": stats["total_pixels"],
        "correct_pixels": stats["correct_pixels"],
        "confusion_matrix_csv": str(csv_path),
        "matrix": stats["matrix"].astype(int).tolist(),
        "classes": stats["classes"],
    }
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump({"validation_results": [result]}, handle, indent=2)
    summary = [
        f"OA={result['overall_accuracy']:.4f}",
        f"Kappa={result['kappa']:.4f}",
        f"Precision={result['macro_precision']:.4f}",
        f"Recall={result['macro_recall']:.4f}",
    ]
    return {
        "report_path": str(report_path),
        "results": [result],
        "summary_lines": summary,
    }


def run_prediction(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_ds, ref_arr, _ref_nodata, _ref_path = _reference_from_config(config)
    boundary_source = config.get("boundary", {}).get("layer_source")
    boundary_mask = _boundary_mask(boundary_source, ref_ds) if boundary_source else np.ones(ref_arr.shape, dtype=bool)

    model_settings = _model_settings(config)
    transitions = config.get("transitions", [])
    annual_rates, train_count = _training_rates(config, boundary_mask, ref_ds)
    if train_count == 0:
        raise RuntimeError("No TRAIN periods were found, so the prediction cannot run.")

    rf_models, feature_importance, rf_notes = _fit_rf_models(config, ref_ds, boundary_mask, transitions)

    latest_period = _latest_observed_period(config)
    base_year = int(latest_period.get("end_year"))
    _, base_array, base_nodata = _align_array(latest_period["end_layer_source"], ref_ds, resample_alg=gdal.GRA_NearestNeighbour)
    valid_mask = _valid_mask(base_array, base_nodata, boundary_mask)

    outputs = []
    enabled_scenarios = [scenario for scenario in config.get("scenarios", []) if scenario.get("enabled")]
    targets = sorted(set(int(year) for year in config.get("target_years", [])))
    if not enabled_scenarios:
        raise RuntimeError("No enabled scenarios were found.")
    if not targets:
        raise RuntimeError("No target years were found.")

    run_report = {
        "project": config.get("project", {}).get("name", ""),
        "base_year": base_year,
        "train_period_count": train_count,
        "annual_transition_rates": annual_rates,
        "outputs": [],
        "rf_notes": rf_notes,
        "model_settings": {
            "use_neighborhood": model_settings.get("use_neighborhood", True),
            "neighborhood_strength": model_settings.get("neighborhood_strength", "Medium"),
        },
    }

    feature_importance_path = output_dir / "feature_importance.json"
    feature_importance_csv_path = output_dir / "feature_importance.csv"
    if feature_importance:
        _write_feature_importance_json(feature_importance_path, feature_importance)
        _write_feature_importance_csv(feature_importance_csv_path, feature_importance, rf_models)
        run_report["feature_importance_json"] = str(feature_importance_path)
        run_report["feature_importance_csv"] = str(feature_importance_csv_path)

    for scenario in enabled_scenarios:
        raw_multipliers = scenario.get("transition_multipliers", {})
        scenario_name = scenario.get("name", "scenario").strip() or "scenario"
        safe_scenario = "".join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in scenario_name.replace(' ', '_'))
        for target_year in targets:
            if target_year <= base_year:
                continue
            suitability_payload = _suitability_maps_for_year(config, ref_ds, boundary_mask, rf_models, target_year=target_year)
            predicted, changed_any, confidence = _predict_to_year(
                base_array,
                base_year,
                target_year,
                valid_mask,
                suitability_payload["fallback"],
                suitability_payload["suitability_by_transition"],
                transitions,
                annual_rates,
                raw_multipliers,
                model_settings,
            )
            predicted_out = predicted.copy().astype(np.int16)
            predicted_out[~valid_mask] = -9999
            out_path = output_dir / f"{config.get('project', {}).get('name', 'project')}_{safe_scenario}_{target_year}.tif"
            _write_raster(out_path, ref_ds, predicted_out, nodata=-9999, gdal_type=gdal.GDT_Int16)
            outputs.append(str(out_path))

            change_map = changed_any.astype(np.int16)
            change_map[~valid_mask] = -9999
            change_path = output_dir / f"{config.get('project', {}).get('name', 'project')}_{safe_scenario}_{target_year}_change.tif"
            _write_raster(change_path, ref_ds, change_map, nodata=-9999, gdal_type=gdal.GDT_Int16)
            outputs.append(str(change_path))

            confidence_map = confidence.astype(np.float32)
            confidence_map[~valid_mask] = -9999.0
            confidence_path = output_dir / f"{config.get('project', {}).get('name', 'project')}_{safe_scenario}_{target_year}_confidence.tif"
            _write_raster(confidence_path, ref_ds, confidence_map, nodata=-9999.0, gdal_type=gdal.GDT_Float32)
            outputs.append(str(confidence_path))

            run_report["outputs"].append({
                "scenario": scenario_name,
                "target_year": target_year,
                "raster": str(out_path),
                "change_raster": str(change_path),
                "confidence_raster": str(confidence_path),
            })

    validation = _run_validation(config, output_dir, ref_ds, boundary_mask, annual_rates, model_settings, rf_models)
    if validation is not None:
        run_report["validation_report"] = validation.get("report_path")
        run_report["validation_results"] = [{
            "scenario": result["scenario"],
            "overall_accuracy": result["overall_accuracy"],
            "kappa": result["kappa"],
            "macro_precision": result["macro_precision"],
            "macro_recall": result["macro_recall"],
            "confusion_matrix_csv": result["confusion_matrix_csv"],
        } for result in validation.get("results", [])]

    report_path = output_dir / f"{config.get('project', {}).get('name', 'project')}_run_report.json"
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(run_report, handle, indent=2)

    log_path = output_dir / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("Predictor2026 v10.5 run summary\n")
        handle.write(f"Base year: {base_year}\n")
        handle.write(f"Training periods used: {train_count}\n")
        handle.write(f"Neighborhood effect: {'On' if model_settings.get('use_neighborhood', True) else 'Off'}\n")
        handle.write(f"Neighborhood strength: {model_settings.get('neighborhood_strength', 'Medium')}\n")
        handle.write("Annual transition rates:\n")
        for code, value in annual_rates.items():
            handle.write(f"- {code}: {value:.6f}\n")
        if rf_notes:
            handle.write("RF notes:\n")
            for note in rf_notes:
                handle.write(f"- {note}\n")
        if feature_importance:
            handle.write(f"Feature importance JSON: {feature_importance_path}\n")

    return {
        "output_dir": str(output_dir),
        "base_year": base_year,
        "train_period_count": train_count,
        "rasters": outputs,
        "report_path": str(report_path),
        "validation": validation,
        "model_settings": {
            "use_neighborhood": model_settings.get("use_neighborhood", True),
            "neighborhood_strength": model_settings.get("neighborhood_strength", "Medium"),
        },
        "feature_importance_path": str(feature_importance_path) if feature_importance else "",
        "feature_importance_csv_path": str(feature_importance_csv_path) if feature_importance else "",
        "log_path": str(log_path),
        "rf_notes": rf_notes,
    }
