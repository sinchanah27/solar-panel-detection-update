# Solar Rooftop Detection Model Card

## Overview
Two models:
- YOLOv8 Detector (bbox)
- YOLOv8 Segmentation (mask)

Purpose: Identify rooftop solar panels for PM Surya Ghar Yojana.

## Architecture
- YOLOv8s backbone for detection
- YOLOv8s-seg for segmentation

## Training Data
- Sources: Satellite imagery (Mapbox, Google Static, OpenAerial)
- Annotation format: YOLO txt + masks
- Class: `solar_panel`

## Metrics
- mAP_50
- mAP_50:95
- Segmentation IoU
- Area estimation error (m²)

## Intended Use
- Rooftop solar verification
- Solar potential assessment

## Limitations
- Cloud cover
- Low resolution tiles
- Incorrect coordinates
- Commercial-level farms

## Ethical Considerations
- No surveillance use
- Not for real-time monitoring
