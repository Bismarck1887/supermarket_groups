# app.py
import os
import time
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from joblib import Parallel, delayed
import tempfile

app = FastAPI()
model = YOLO('bestY8.pt')


def rect_to_polygon(row):
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def run_yolo_inference_array(img_array):
    result = model(img_array, verbose=False)[0]
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    return {
        'detections': list(zip(xyxy, cls))
    }


def group_boxes(detections):
    data = []
    for xc in detections:
        box, class_id = xc
        data.append({
            'class_id': int(class_id),
            'x1': int(box[0]),
            'y1': int(box[1]),
            'x2': int(box[2]),
            'y2': int(box[3]),
            'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        })

    df = pd.DataFrame(data)
    df['geometry'] = df.apply(rect_to_polygon, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf['square'] = gdf.geometry.area

    group_gdf = gdf.copy()
    group_gdf['group'] = np.nan
    group_id = 1

    id_list = []

    def assign_group(row):
        nonlocal group_id
        nonlocal id_list
        if row['class_id'] in [0, 2]:
            row['group'] = group_id
            group_id += 1
            id_list.append(row.name)
        return row['group']

    group_gdf['group'] = group_gdf.apply(assign_group, axis=1)

    unlabeled = group_gdf[group_gdf['class_id'] == 1].copy()
    grouped = group_gdf.loc[id_list][['geometry', 'group']].copy()
    next_group_id = group_id

    def find_best_group_grouped(current_geom, grouped):
        intersection_areas = grouped['geometry'].apply(
            lambda geom: current_geom.intersection(geom).area if not current_geom.intersection(geom).is_empty else 0)
        nonlocal next_group_id
        if intersection_areas.max() > 0:
            best_match_idx = intersection_areas.idxmax()
            return grouped.loc[best_match_idx, 'group']
        else:
            next_group_id += 1
            return next_group_id - 1

    unlabeled['group'] = unlabeled['geometry'].apply(lambda geom: find_best_group_grouped(geom, grouped))
    group_gdf.update(unlabeled[['group']])

    group_counts = group_gdf['group'].value_counts()
    single_occurrence_groups = group_counts[group_counts == 1].index
    condition_to_drop = group_gdf['group'].isin(single_occurrence_groups) & group_gdf['class_id'].isin([0, 2])
    group_gdf = group_gdf[~condition_to_drop]

    return group_gdf['group'].nunique()


@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    start_gpu = time.time()
    # Запуск инференса параллельно, если нужно обработать несколько изображений
    detection_info = run_yolo_inference_array(img)
    end_gpu = time.time()

    start_cpu = time.time()
    group_count = group_boxes(detection_info['detections'])
    end_cpu = time.time()

    return JSONResponse(content={
        "filename": file.filename,
        "GPU_inference_time_sec": round(end_gpu - start_gpu, 2),
        "CPU_grouping_time_sec": round(end_cpu - start_cpu, 2),
        "group_count": group_count
    })


@app.post("/process_multiple_images/")
async def process_multiple_images(files: list[UploadFile] = File(...)):

    # Чтение всех файлов
    images = []
    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        images.append(img)

    # Распараллеливаем выполнение инференса для разных изображений
    start_gpu = time.time()
    detections = Parallel(n_jobs=-1)(
        delayed(run_yolo_inference_array)(img) for img in images
    )
    end_gpu = time.time()

    # Группируем все изображения
    start_cpu = time.time()
    grouped_counts = [group_boxes(det['detections']) for det in detections]
    end_cpu = time.time()

    # Вычисление среднего времени для каждого изображения
    total_objects = sum([len(det['detections']) for det in detections])
    avg_gpu_time_per_object = round((end_gpu - start_gpu) / total_objects, 4) if total_objects > 0 else 0
    avg_cpu_time_per_object = round((end_cpu - start_cpu) / total_objects, 4) if total_objects > 0 else 0

    return JSONResponse(content={
        "GPU_inference_time_sec": round(end_gpu - start_gpu, 2),
        "CPU_grouping_time_sec": round(end_cpu - start_cpu, 2),
        "group_counts": grouped_counts,
        "avg_GPU_time_per_object_sec": avg_gpu_time_per_object,
        "avg_CPU_time_per_object_sec": avg_cpu_time_per_object
    })