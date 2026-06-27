-- MySQL DDL Schema for ANPR Detections Database
-- Table: detections

CREATE DATABASE IF NOT EXISTS anpr_db;
USE anpr_db;

CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid VARCHAR(36) NOT NULL UNIQUE,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    vehicle_type VARCHAR(50) DEFAULT NULL,
    plate_number VARCHAR(20) DEFAULT NULL,
    vehicle_confidence FLOAT DEFAULT NULL,
    plate_confidence FLOAT DEFAULT NULL,
    ocr_confidence FLOAT DEFAULT NULL,
    processing_time_ms FLOAT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    annotated_image_path VARCHAR(500) DEFAULT NULL,
    json_result TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
