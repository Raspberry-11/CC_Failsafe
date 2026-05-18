-- Active: 1779015058740@@127.0.0.1@3306
-- Run this in XAMPP phpMyAdmin or MySQL terminal before starting the backend

CREATE DATABASE IF NOT EXISTS failsafe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE failsafe;

-- Tables are auto-created by SQLAlchemy on first run.
-- This file just ensures the database exists.

SELECT 'Database "failsafe" ready.' AS status;
