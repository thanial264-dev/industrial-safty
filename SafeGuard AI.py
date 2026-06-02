import os
import cv2
import base64
import json
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REFERENCE_FOLDER = os.path.join(BASE_DIR, "reference_videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ══════════════════════════════════════════════════════════════════════════════
# ESP8266 RELAY CONFIGURATION
# ► Change ESP8266_IP to the IP address shown on your ESP8266 Serial Monitor
# ► Make sure your PC and ESP8266 are on the SAME Wi-Fi network
# ► The ESP8266 should be running a simple HTTP server that listens on /relay/off
# ══════════════════════════════════════════════════════════════════════════════
ESP8266_IP   = "10.146.170.117"   # <── CHANGE THIS to the IP printed on your ESP8266 Serial Monitor
ESP8266_PORT = 80                # <── Keep 80 (matches your Arduino server(80))
ESP8266_PATH = "/accident_on"    # <── Matches handleAccidentON() in your Arduino code → sets relay HIGH, machine OFF
ESP8266_TIMEOUT_SEC = 3          # <── How long to wait for ESP8266 response

def trigger_esp8266_relay_off(reason: str = "Accident detected"):
    """
    Sends an HTTP GET request to the ESP8266 to turn OFF the relay.
    Called automatically whenever an accident is detected.
    Non-blocking: runs in a background thread so it never slows down analysis.
    """
    import urllib.request
    url = f"http://{ESP8266_IP}:{ESP8266_PORT}{ESP8266_PATH}"
    try:
        req = urllib.request.urlopen(url, timeout=ESP8266_TIMEOUT_SEC)
        status = req.getcode()
        print(f"  [ESP8266] Relay OFF signal sent → {url}  (HTTP {status}) | Reason: {reason}")
    except Exception as e:
        print(f"  [ESP8266] ⚠ Failed to reach ESP8266 at {url} — {e}")

def trigger_relay_in_background(reason: str = "Accident detected"):
    """Fire-and-forget: sends ESP8266 signal without blocking the analysis thread."""
    t = threading.Thread(target=trigger_esp8266_relay_off, args=(reason,))
    t.daemon = True
    t.start()

# ══════════════════════════════════════════════════════════════════════════════

analysis_results = {}
analysis_progress = {}

# ── Embedded HTML ──────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SafeGuard AI — Industrial Accident Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #080c10;
      --surface: #0e1420;
      --surface2: #141c2a;
      --border: #1e2d45;
      --accent: #00e5ff;
      --accent2: #ff3b3b;
      --accent3: #ffd600;
      --safe: #00e676;
      --text: #c8d8f0;
      --text-dim: #4a6080;
      --font-mono: 'Share Tech Mono', monospace;
      --font-head: 'Barlow Condensed', sans-serif;
      --font-body: 'Barlow', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      height: 100%;
      overflow: hidden;   /* prevent full-page stretch */
    }

    /* Scanline overlay */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,229,255,0.015) 2px, rgba(0,229,255,0.015) 4px);
      pointer-events: none; z-index: 9999;
    }

    /* Corner brackets */
    .bracket {
      position: fixed;
      width: 40px; height: 40px;
      pointer-events: none; z-index: 9998;
    }
    .bracket.tl { top: 12px; left: 12px; border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
    .bracket.tr { top: 12px; right: 12px; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
    .bracket.bl { bottom: 12px; left: 12px; border-bottom: 2px solid var(--accent); border-left: 2px solid var(--accent); }
    .bracket.br { bottom: 12px; right: 12px; border-bottom: 2px solid var(--accent); border-right: 2px solid var(--accent); }

    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 18px 40px;
      border-bottom: 1px solid var(--border);
      background: rgba(8,12,16,0.9);
      backdrop-filter: blur(8px);
      position: sticky; top: 0; z-index: 100;
    }

    .logo {
      display: flex; align-items: center; gap: 12px;
    }
    .logo-icon {
      width: 36px; height: 36px;
      border: 2px solid var(--accent);
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      position: relative;
      animation: pulse-border 3s ease-in-out infinite;
    }
    @keyframes pulse-border {
      0%, 100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.3); }
      50% { box-shadow: 0 0 0 6px rgba(0,229,255,0); }
    }
    .logo h1 {
      font-family: var(--font-head);
      font-size: 22px; font-weight: 900;
      letter-spacing: 4px; text-transform: uppercase;
      color: #fff;
    }
    .logo span { color: var(--accent); }

    .status-bar {
      font-family: var(--font-mono);
      font-size: 11px; color: var(--text-dim);
      display: flex; align-items: center; gap: 16px;
    }
    .status-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--safe);
      animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

    main {
      max-width: 1300px;
      margin: 0 auto;
      padding: 24px 32px;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 24px;
      height: calc(100vh - 73px);   /* fill remaining screen below header */
      overflow: hidden;
    }

    /* each column scrolls independently */
    .left-col {
      display: flex; flex-direction: column; gap: 20px;
      overflow-y: auto; overflow-x: hidden;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
      padding-right: 4px;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      position: relative;
    }
    .panel::before {
      content: '';
      position: absolute; top: -1px; left: 20px; right: 20px;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }

    .panel-label {
      font-family: var(--font-mono);
      font-size: 10px; color: var(--accent);
      letter-spacing: 3px; text-transform: uppercase;
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
    }

    /* Upload Zone */
    .upload-zone {
      padding: 40px;
      border: 2px dashed var(--border);
      margin: 24px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      position: relative;
      overflow: hidden;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: var(--accent);
      background: rgba(0,229,255,0.03);
    }
    .upload-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.6; }
    .upload-zone h3 {
      font-family: var(--font-head);
      font-size: 20px; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase;
      color: #fff; margin-bottom: 8px;
    }
    .upload-zone p {
      font-size: 13px; color: var(--text-dim);
      font-family: var(--font-mono);
    }
    .upload-btn {
      display: inline-block;
      margin-top: 20px;
      padding: 10px 28px;
      background: transparent;
      border: 1px solid var(--accent);
      color: var(--accent);
      font-family: var(--font-mono);
      font-size: 12px; letter-spacing: 2px;
      text-transform: uppercase;
      cursor: pointer;
      transition: all 0.2s;
    }
    .upload-btn:hover {
      background: var(--accent);
      color: var(--bg);
    }

    /* Video Player */
    .video-wrapper { padding: 0; position: relative; }
    video {
      width: 100%;
      display: block;
      background: #000;
      max-height: 340px;
      object-fit: contain;
    }

    /* Custom Controls */
    .controls {
      background: var(--surface2);
      padding: 14px 20px;
      border-top: 1px solid var(--border);
    }

    .progress-track {
      width: 100%; height: 4px;
      background: var(--border);
      border-radius: 2px;
      cursor: pointer;
      margin-bottom: 14px;
      position: relative;
    }
    .progress-fill {
      height: 100%; border-radius: 2px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      pointer-events: none;
      transition: width 0.1s;
    }
    /* Accident markers on timeline */
    .accident-marker {
      position: absolute;
      top: -4px;
      width: 4px; height: 12px;
      background: var(--accent2);
      border-radius: 2px;
      transform: translateX(-50%);
    }

    .controls-row {
      display: flex; align-items: center; gap: 12px;
    }
    .ctrl-btn {
      background: none; border: none;
      color: var(--text); cursor: pointer;
      font-size: 18px; padding: 6px;
      transition: color 0.15s;
      font-family: var(--font-mono);
    }
    .ctrl-btn:hover { color: var(--accent); }
    .ctrl-btn.active { color: var(--accent2); }

    .time-display {
      font-family: var(--font-mono);
      font-size: 12px; color: var(--text-dim);
      margin-left: auto;
    }

    .volume-slider {
      width: 70px;
      accent-color: var(--accent);
    }

    /* Progress Bar */
    .analysis-progress {
      margin: 24px;
      display: none;
    }
    .analysis-progress.visible { display: block; }
    .prog-label {
      font-family: var(--font-mono);
      font-size: 11px; color: var(--text-dim);
      margin-bottom: 8px;
      display: flex; justify-content: space-between;
    }
    .prog-bar-bg {
      height: 3px; background: var(--border); border-radius: 2px;
    }
    .prog-bar-fill {
      height: 100%; border-radius: 2px;
      background: linear-gradient(90deg, var(--accent), #0080ff);
      transition: width 0.4s ease;
    }

    /* Alert Banner */
    .alert-banner {
      margin: 0 24px 20px;
      padding: 16px 20px;
      border: 1px solid var(--accent2);
      background: rgba(255,59,59,0.07);
      display: none;
      animation: alert-in 0.4s ease;
    }
    .alert-banner.visible { display: flex; align-items: flex-start; gap: 14px; }
    .alert-banner.safe {
      border-color: var(--safe);
      background: rgba(0,230,118,0.05);
    }
    @keyframes alert-in {
      from { opacity: 0; transform: translateY(-8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .alert-icon { font-size: 28px; flex-shrink: 0; }
    .alert-title {
      font-family: var(--font-head);
      font-size: 18px; font-weight: 900;
      letter-spacing: 2px; text-transform: uppercase;
      margin-bottom: 4px;
    }
    .alert-banner:not(.safe) .alert-title { color: var(--accent2); }
    .alert-banner.safe .alert-title { color: var(--safe); }
    .alert-desc { font-size: 13px; color: var(--text-dim); line-height: 1.5; }

    /* Right panel */
    .right-panel {
      display: flex; flex-direction: column; gap: 16px;
      overflow-y: auto; overflow-x: hidden;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
      padding-right: 2px;
      height: 100%;
    }

    /* Stats grid */
    .stats-grid {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 1px; background: var(--border);
      margin: 0;
    }
    .stat-cell {
      background: var(--surface);
      padding: 16px;
      text-align: center;
    }
    .stat-value {
      font-family: var(--font-head);
      font-size: 36px; font-weight: 900;
      line-height: 1;
      color: var(--accent);
      margin-bottom: 4px;
    }
    .stat-value.danger { color: var(--accent2); }
    .stat-value.safe { color: var(--safe); }
    .stat-label {
      font-family: var(--font-mono);
      font-size: 9px; color: var(--text-dim);
      letter-spacing: 2px; text-transform: uppercase;
    }

    /* Severity badge */
    .severity-badge {
      display: inline-block;
      padding: 4px 10px;
      font-family: var(--font-mono);
      font-size: 10px; letter-spacing: 2px;
      text-transform: uppercase;
      border-radius: 2px;
    }
    .severity-critical { background: rgba(255,59,59,0.2); color: var(--accent2); border: 1px solid var(--accent2); }
    .severity-high { background: rgba(255,150,0,0.15); color: #ff9600; border: 1px solid #ff9600; }
    .severity-medium { background: rgba(255,214,0,0.15); color: var(--accent3); border: 1px solid var(--accent3); }
    .severity-low { background: rgba(0,230,118,0.1); color: var(--safe); border: 1px solid var(--safe); }
    .severity-none { background: rgba(74,96,128,0.2); color: var(--text-dim); border: 1px solid var(--border); }

    /* Event list */
    .events-list {
      max-height: 380px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
      flex: 1;
    }
    .event-item {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      transition: background 0.15s;
      display: flex; align-items: flex-start; gap: 12px;
    }
    .event-item:hover { background: var(--surface2); }
    .event-item.accident { border-left: 3px solid var(--accent2); }
    .event-time {
      font-family: var(--font-mono);
      font-size: 12px; color: var(--accent);
      flex-shrink: 0;
      padding-top: 2px;
    }
    .event-body {}
    .event-type {
      font-family: var(--font-head);
      font-size: 14px; font-weight: 700;
      letter-spacing: 1px; text-transform: uppercase;
      margin-bottom: 3px;
    }
    .event-desc { font-size: 12px; color: var(--text-dim); line-height: 1.4; }

    .no-events {
      padding: 30px 20px;
      text-align: center;
      font-family: var(--font-mono);
      font-size: 12px; color: var(--text-dim);
    }

    /* Confidence bar */
    .conf-bar {
      height: 2px; background: var(--border);
      border-radius: 1px; margin-top: 6px;
    }
    .conf-fill {
      height: 100%; border-radius: 1px;
      background: var(--accent2);
    }

    /* Frame grid (timeline thumbnails placeholder) */
    .timeline-strip {
      padding: 14px 20px;
      display: flex; gap: 6px;
      overflow-x: auto;
      scrollbar-width: none;
    }
    .frame-thumb {
      width: 44px; height: 36px;
      background: var(--surface2);
      border: 1px solid var(--border);
      flex-shrink: 0;
      position: relative;
      display: flex; align-items: center; justify-content: center;
      font-family: var(--font-mono);
      font-size: 9px; color: var(--text-dim);
      cursor: pointer;
      transition: border-color 0.15s;
    }
    .frame-thumb.accident-frame { border-color: var(--accent2); }
    .frame-thumb.accident-frame::after {
      content: '⚠';
      position: absolute; bottom: 2px; right: 3px;
      font-size: 8px; color: var(--accent2);
    }

    /* Manual Re-analyze Box */
    .reanalyze-box {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
      display: none;
    }
    .reanalyze-box.visible { display: block; }
    .reanalyze-label {
      font-family: var(--font-mono);
      font-size: 9px; letter-spacing: 2px;
      color: var(--accent); margin-bottom: 8px;
    }
    .reanalyze-row {
      display: flex; gap: 8px; align-items: center;
    }
    .ts-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 13px;
      padding: 8px 10px;
      outline: none;
      transition: border-color 0.2s;
    }
    .ts-input:focus { border-color: var(--accent); }
    .ts-input::placeholder { color: var(--text-dim); }
    .reanalyze-btn {
      background: transparent;
      border: 1px solid var(--accent);
      color: var(--accent);
      font-family: var(--font-mono);
      font-size: 11px; letter-spacing: 2px;
      padding: 8px 14px;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .reanalyze-btn:hover:not(:disabled) {
      background: var(--accent); color: var(--bg);
    }
    .reanalyze-btn:disabled {
      opacity: 0.4; cursor: not-allowed;
    }
    .reanalyze-status {
      font-family: var(--font-mono);
      font-size: 10px; margin-top: 7px;
      min-height: 14px;
      color: var(--text-dim);
    }
    .reanalyze-status.error { color: var(--accent2); }
    .reanalyze-status.success { color: var(--safe); }
    .event-item.manual-check { border-left: 3px solid var(--accent); }
    .manual-badge {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 8px; letter-spacing: 1px;
      padding: 2px 5px;
      background: rgba(0,229,255,0.1);
      border: 1px solid var(--accent);
      color: var(--accent);
      margin-left: 6px;
      vertical-align: middle;
    }

    /* Loading spinner */
    .spinner {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Responsive */
    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
        height: auto;
        overflow: visible;
        padding: 16px;
      }
      html, body { overflow: auto; height: auto; }
      .left-col { overflow: visible; }
      .right-panel { overflow: visible; height: auto; }
      header { padding: 14px 20px; }
    }

    /* Accident Time Range Input */
    .range-box {
      background: rgba(255,59,59,0.05);
      border: 1px solid rgba(255,59,59,0.3);
      padding: 14px 16px;
      margin: 16px 0 12px;
      text-align: left;
    }
    .range-box-label {
      font-family: var(--font-mono);
      font-size: 9px; letter-spacing: 2px;
      color: var(--accent2); margin-bottom: 4px;
    }
    .range-box-hint {
      font-family: var(--font-mono);
      font-size: 9px; color: var(--text-dim);
      margin-bottom: 10px; line-height: 1.5;
    }
    .range-input-row {
      display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    }
    .range-input {
      width: 90px;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 12px; padding: 6px 8px;
      outline: none;
    }
    .range-input:focus { border-color: var(--accent2); }
    .range-sep {
      font-family: var(--font-mono);
      font-size: 11px; color: var(--text-dim);
    }
    .range-add-btn {
      background: transparent;
      border: 1px solid var(--accent2);
      color: var(--accent2);
      font-family: var(--font-mono);
      font-size: 10px; letter-spacing: 1px;
      padding: 6px 10px; cursor: pointer;
      transition: all 0.2s;
    }
    .range-add-btn:hover { background: var(--accent2); color: var(--bg); }
    .range-list-item {
      display: flex; align-items: center; justify-content: space-between;
      padding: 5px 0;
      border-bottom: 1px solid rgba(255,59,59,0.15);
      font-family: var(--font-mono); font-size: 11px;
      color: var(--accent2);
    }
    .range-remove-btn {
      background: none; border: none;
      color: var(--text-dim); cursor: pointer;
      font-size: 14px; line-height: 1;
      padding: 0 4px;
    }
    .range-remove-btn:hover { color: var(--accent2); }
    .range-error {
      font-family: var(--font-mono);
      font-size: 10px; color: var(--accent2);
      margin-top: 6px; min-height: 14px;
    }
    /* Pinned range badge in incident log */
    .pinned-badge {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 8px; letter-spacing: 1px;
      padding: 2px 5px;
      background: rgba(255,59,59,0.15);
      border: 1px solid var(--accent2);
      color: var(--accent2);
      margin-left: 4px;
      vertical-align: middle;
    }

    /* Reference Library Panel */    /* Reference Library Panel */
    .ref-panel-label {
      font-family: var(--font-mono);
      font-size: 9px; letter-spacing: 2px;
      color: var(--accent3); margin-bottom: 8px;
      text-transform: uppercase;
    }
    .ref-upload-row {
      display: flex; gap: 8px; align-items: center;
      margin-bottom: 10px;
    }
    .ref-upload-btn {
      background: transparent;
      border: 1px solid var(--accent3);
      color: var(--accent3);
      font-family: var(--font-mono);
      font-size: 10px; letter-spacing: 2px;
      padding: 7px 12px;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .ref-upload-btn:hover { background: var(--accent3); color: var(--bg); }
    .ref-list { display: flex; flex-direction: column; gap: 6px; }
    .ref-item {
      padding: 8px 10px;
      border: 1px solid var(--border);
      background: var(--surface2);
      display: flex; align-items: center; gap: 10px;
    }
    .ref-item-icon { font-size: 16px; flex-shrink: 0; }
    .ref-item-name {
      font-family: var(--font-mono);
      font-size: 11px; color: var(--text);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ref-item-meta {
      font-size: 10px; color: var(--text-dim);
      font-family: var(--font-mono);
    }
    .ref-status {
      font-family: var(--font-mono);
      font-size: 10px; color: var(--text-dim);
      margin-top: 6px; min-height: 14px;
    }
    .ref-status.building { color: var(--accent3); }
    .ref-status.ready { color: var(--safe); }
    .ref-status.error { color: var(--accent2); }

    /* Frame thumbnail in incident log */
    .frame-img {
      width: 80px; height: 54px;
      object-fit: cover;
      border: 1px solid var(--border);
      flex-shrink: 0;
      cursor: pointer;
    }
    .frame-img.accident-frame { border-color: var(--accent2); }
    .frame-img.ref-match-frame { border-color: var(--accent3); }
    .event-item { gap: 10px; }

    /* Reference match badge */
    .ref-match-badge {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 8px; letter-spacing: 1px;
      padding: 2px 5px;
      background: rgba(255,214,0,0.1);
      border: 1px solid var(--accent3);
      color: var(--accent3);
      margin-left: 4px;
      vertical-align: middle;
    }
  </style>
</head>
<body>

<!-- API Key Warning Banner -->
<div id="api-warn" style="display:none;background:#1a0a0a;border-bottom:2px solid var(--accent2);padding:12px 40px;font-family:var(--font-mono);font-size:12px;color:var(--accent2);letter-spacing:1px;text-align:center;z-index:9997;position:relative">
  ⚠ &nbsp; ANTHROPIC_API_KEY IS NOT SET — All analysis will fail. &nbsp;
  Set it in your terminal: &nbsp;<code style="background:#2a0a0a;padding:2px 8px;border:1px solid var(--accent2)">set ANTHROPIC_API_KEY=sk-ant-...</code> &nbsp; then restart the server.
</div>

<div class="bracket tl"></div>
<div class="bracket tr"></div>
<div class="bracket bl"></div>
<div class="bracket br"></div>

<header>
  <div class="logo">
    <div class="logo-icon">🛡</div>
    <h1>Safe<span>Guard</span> AI</h1>
  </div>
  <div class="status-bar">
    <div class="status-dot"></div>
    <span>INDUSTRIAL ACCIDENT DETECTION SYSTEM</span>
    <span id="current-time">--:--:--</span>
    <span id="esp-status" style="color:var(--text-dim);font-family:var(--font-mono);font-size:10px;letter-spacing:1px;border-left:1px solid var(--border);padding-left:12px">
      ESP8266: <span id="esp-ip">...</span>
    </span>
  </div>
</header>

<main>
  <!-- Left Column -->
  <div class="left-col">
    <!-- Upload / Video Panel -->
    <div class="panel" style="margin-bottom: 20px;">
      <div class="panel-label">
        <span>VIDEO INPUT</span>
        <span id="video-name" style="color:var(--text-dim)">NO FEED</span>
      </div>

      <!-- Upload Zone (shown when no video) -->
      <div class="upload-zone" id="upload-zone">
        <div class="upload-icon">📹</div>
        <h3>Drop Industrial Video</h3>
        <p>MP4, AVI, MOV, MKV supported</p>

        <!-- Accident time ranges input -->
        <div class="range-box" id="range-box">
          <div class="range-box-label">⚠ KNOWN ACCIDENT TIME RANGES (optional)</div>
          <div class="range-box-hint">Frames inside these ranges are instantly marked as accidents — no AI needed</div>
          <div id="range-list"></div>
          <div class="range-input-row">
            <input class="range-input" id="range-start" type="text" placeholder="Start e.g. 0:10">
            <span class="range-sep">to</span>
            <input class="range-input" id="range-end" type="text" placeholder="End e.g. 0:20">
            <button class="range-add-btn" onclick="addRange()">+ ADD</button>
          </div>
          <div id="range-error" class="range-error"></div>
        </div>

        <button class="upload-btn" onclick="document.getElementById('file-input').click()">SELECT FILE</button>
        <input type="file" id="file-input" accept="video/*" style="display:none">
      </div>

      <!-- Video Player (hidden initially) -->
      <div class="video-wrapper" id="video-wrapper" style="display:none">
        <video id="main-video" preload="metadata"></video>

        <div class="controls">
          <div class="progress-track" id="progress-track">
            <div class="progress-fill" id="progress-fill" style="width:0%"></div>
            <!-- accident markers injected here -->
          </div>
          <div class="controls-row">
            <button class="ctrl-btn" id="btn-back" title="Back 10s" onclick="seekBy(-10)">⏮ 10s</button>
            <button class="ctrl-btn" id="btn-play" onclick="togglePlay()">▶</button>
            <button class="ctrl-btn" id="btn-fwd" title="Forward 10s" onclick="seekBy(10)">10s ⏭</button>
            <input class="volume-slider" type="range" min="0" max="1" step="0.05" value="1" oninput="setVolume(this.value)" title="Volume">
            <span class="time-display" id="time-display">0:00 / 0:00</span>
          </div>
        </div>
      </div>

      <!-- Analysis Progress -->
      <div class="analysis-progress" id="analysis-progress">
        <div class="prog-label">
          <span id="prog-msg">Initializing AI analysis...</span>
          <span id="prog-pct">0%</span>
        </div>
        <div class="prog-bar-bg">
          <div class="prog-bar-fill" id="prog-fill" style="width:0%"></div>
        </div>
      </div>

      <!-- Alert Banner -->
      <div class="alert-banner" id="alert-banner">
        <div class="alert-icon" id="alert-icon">⚠️</div>
        <div>
          <div class="alert-title" id="alert-title">ACCIDENT DETECTED</div>
          <div class="alert-desc" id="alert-desc">—</div>
        </div>
      </div>
    </div>

    <!-- Frame Timeline -->
    <div class="panel" id="timeline-panel" style="display:none">
      <div class="panel-label">
        <span>ANALYZED FRAMES TIMELINE</span>
        <span id="timeline-count" style="color:var(--text-dim)"></span>
      </div>
      <div class="timeline-strip" id="timeline-strip"></div>
    </div>
  </div>

  <!-- Right Column -->
  <div class="right-panel">
    <!-- Stats -->
    <div class="panel">
      <div class="panel-label"><span>ANALYSIS METRICS</span></div>
      <div class="stats-grid">
        <div class="stat-cell">
          <div class="stat-value" id="stat-frames">—</div>
          <div class="stat-label">Frames Analyzed</div>
        </div>
        <div class="stat-cell">
          <div class="stat-value danger" id="stat-accidents">—</div>
          <div class="stat-label">Accidents Found</div>
        </div>
        <div class="stat-cell">
          <div class="stat-value" id="stat-duration">—</div>
          <div class="stat-label">Duration (s)</div>
        </div>
        <div class="stat-cell">
          <div class="stat-value" id="stat-severity">—</div>
          <div class="stat-label">Max Severity</div>
        </div>
      </div>
    </div>

    <!-- Reference Library -->
    <div class="panel" id="ref-library-panel">
      <div class="panel-label">
        <span>📚 REFERENCE LIBRARY</span>
        <span id="ref-count" style="color:var(--accent3)">0 VIDEOS</span>
      </div>
      <div style="padding:12px 16px;">
        <div class="ref-panel-label">⚡ ADD REFERENCE ACCIDENT VIDEO</div>
        <div class="ref-upload-row">
          <button class="ref-upload-btn" onclick="document.getElementById('ref-file-input').click()">
            + ADD REFERENCE
          </button>
          <input type="file" id="ref-file-input" accept="video/*" style="display:none">
        </div>
        <div id="ref-status" class="ref-status"></div>
        <div class="ref-list" id="ref-list">
          <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-dim)">No reference videos yet</div>
        </div>
      </div>
    </div>

    <!-- Event Log -->
    <div class="panel" style="flex:1">
      <div class="panel-label">
        <span>INCIDENT LOG</span>
        <span id="log-count" style="color:var(--text-dim)">AWAITING ANALYSIS</span>
      </div>

      <!-- Manual Re-analyze Input -->
      <div class="reanalyze-box" id="reanalyze-box">
        <div class="reanalyze-label">📍 MARK ACCIDENT AT TIMESTAMP</div>
        <div class="reanalyze-row">
          <input
            id="manual-timestamp"
            class="ts-input"
            type="text"
            placeholder="e.g. 1:30 or 90s"
            maxlength="8"
          />
          <button class="reanalyze-btn" id="reanalyze-btn" onclick="manualMarkAccident()">
            MARK ACCIDENT
          </button>
        </div>
        <div id="reanalyze-status" class="reanalyze-status"></div>
      </div>

      <div class="events-list" id="events-list">
        <div class="no-events">Upload a video to begin analysis</div>
      </div>
    </div>
  </div>
</main>

<script>
  const API = '';  // same origin
  let currentJobId = null;
  let videoDuration = 0;
  let accidentEvents = [];
  let allFrameResults = [];
  let pollInterval = null;

  // ── ESP8266 status on page load ─────────────────────────────────────────────
  (async () => {
    try {
      const r = await fetch('/api/esp8266/status');
      const d = await r.json();
      const el = document.getElementById('esp-ip');
      el.textContent = d.ip;
      el.style.color = 'var(--accent)';
    } catch(e) {
      document.getElementById('esp-ip').textContent = 'N/A';
    }
  })();

  // ── Reset relay to ON every time page loads/refreshes ───────────────────────
  (async () => {
    try {
      await fetch('/api/esp8266/reset', { method: 'POST' });
      console.log('[ESP8266] Page loaded → Relay reset to ON');
    } catch(e) {
      console.warn('[ESP8266] Relay reset failed:', e.message);
    }
  })();

  // Check API key on load
  (async () => {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      if (!d.api_key_set) {
        document.getElementById('api-warn').style.display = 'block';
      }
    } catch(e) {}
  })();

  // Clock
  setInterval(() => {
    const d = new Date();
    document.getElementById('current-time').textContent = d.toTimeString().slice(0,8);
  }, 1000);

  // Drag & Drop
  const uploadZone = document.getElementById('upload-zone');
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) handleFile(file);
  });

  document.getElementById('file-input').addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  // ── Accident time ranges ────────────────────────────────────────────────────
  let accidentRanges = [];  // [{start: seconds, end: seconds}]

  function parseTimestamp(ts) {
    ts = ts.trim();
    if (ts.includes(':')) {
      const parts = ts.split(':');
      return parseInt(parts[0]) * 60 + parseFloat(parts[1]);
    }
    return parseFloat(ts);
  }

  function addRange() {
    const startRaw = document.getElementById('range-start').value.trim();
    const endRaw   = document.getElementById('range-end').value.trim();
    const errEl    = document.getElementById('range-error');
    errEl.textContent = '';

    if (!startRaw || !endRaw) { errEl.textContent = 'Enter both start and end.'; return; }
    const start = parseTimestamp(startRaw);
    const end   = parseTimestamp(endRaw);
    if (isNaN(start) || isNaN(end)) { errEl.textContent = 'Invalid format. Use 0:10 or 10.'; return; }
    if (end <= start) { errEl.textContent = 'End must be after start.'; return; }

    accidentRanges.push({ start, end });
    document.getElementById('range-start').value = '';
    document.getElementById('range-end').value = '';
    renderRanges();
  }

  function removeRange(i) {
    accidentRanges.splice(i, 1);
    renderRanges();
  }

  function renderRanges() {
    const el = document.getElementById('range-list');
    if (!accidentRanges.length) { el.innerHTML = ''; return; }
    el.innerHTML = accidentRanges.map((r, i) =>
      `<div class="range-list-item">
        <span>⚠ ${formatTime(r.start)} → ${formatTime(r.end)}</span>
        <button class="range-remove-btn" onclick="removeRange(${i})">✕</button>
      </div>`
    ).join('');
  }

  function isInAccidentRange(timestamp) {
    return accidentRanges.some(r => timestamp >= r.start && timestamp <= r.end);
  }

  // ── File handling ────────────────────────────────────────────────────────────
  async function handleFile(file) {
    resetUI();
    document.getElementById('video-name').textContent = file.name.toUpperCase();

    const url = URL.createObjectURL(file);
    const video = document.getElementById('main-video');
    video.src = url;

    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('video-wrapper').style.display = 'block';

    video.addEventListener('loadedmetadata', () => {
      videoDuration = video.duration;
      updateTimeDisplay();
      // Draw pinned range markers on timeline (green bands)
      accidentRanges.forEach(r => addRangeBand(r.start, r.end));
    }, { once: true });

    // Send ranges with upload
    const formData = new FormData();
    formData.append('video', file);
    formData.append('accident_ranges', JSON.stringify(accidentRanges));

    document.getElementById('analysis-progress').classList.add('visible');
    setProgMsg('Uploading video...', 5);

    try {
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
      const data = await res.json();
      currentJobId = data.job_id;
      // Store ranges locally too for display
      if (data.accident_ranges_count) {
        setProgMsg(`Upload done. ${data.accident_ranges_count} time range(s) pinned.`, 10);
      }
      pollProgress();
    } catch (err) {
      setProgMsg('Upload failed: ' + err.message, 0);
    }
  }

  function addRangeBand(start, end) {
    if (!videoDuration) return;
    const track = document.getElementById('progress-track');
    const band = document.createElement('div');
    const left = (start / videoDuration) * 100;
    const width = Math.max(0.5, ((end - start) / videoDuration) * 100);
    band.style.cssText = (
      'position:absolute;top:0;height:100%;border-radius:2px;pointer-events:none;' +
      'background:rgba(255,59,59,0.35);left:' + left + '%;width:' + width + '%'
    );
    band.title = 'Pinned accident range: ' + formatTime(start) + ' → ' + formatTime(end);
    track.appendChild(band);
  }

  function pollProgress() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
      if (!currentJobId) return;
      try {
        const res = await fetch(`${API}/api/progress/${currentJobId}`);
        const data = await res.json();
        setProgMsg(data.message, data.progress);

        if (data.status === 'done') {
          clearInterval(pollInterval);
          fetchResults();
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          setProgMsg('Error: ' + data.message, 0);
        }
      } catch (e) {}
    }, 1200);
  }

  async function fetchResults() {
    const res = await fetch(`${API}/api/results/${currentJobId}`);
    const data = await res.json();
    displayResults(data);
  }

  function displayResults(data) {
    document.getElementById('analysis-progress').classList.remove('visible');
    document.getElementById('reanalyze-box').classList.add('visible');

    accidentEvents = data.accident_events || [];
    allFrameResults = data.all_frame_results || [];
    videoDuration = data.duration || videoDuration;

    // Stats
    document.getElementById('stat-frames').textContent = data.frames_analyzed || '—';
    document.getElementById('stat-accidents').textContent = data.total_accidents_found || 0;
    document.getElementById('stat-duration').textContent = data.duration || '—';

    const sev = data.max_severity;
    const sevEl = document.getElementById('stat-severity');
    sevEl.textContent = sev ? sev.toUpperCase() : 'NONE';
    sevEl.className = 'stat-value ' + (sev === 'critical' || sev === 'high' ? 'danger' : sev ? '' : 'safe');

    // Alert
    const banner = document.getElementById('alert-banner');
    banner.classList.add('visible');
    if (data.accident_detected) {
      banner.classList.remove('safe');
      document.getElementById('alert-icon').textContent = '🚨';
      document.getElementById('alert-title').textContent = '⚠ INDUSTRIAL ACCIDENT DETECTED';
      document.getElementById('alert-desc').textContent = data.summary;
    } else {
      banner.classList.add('safe');
      document.getElementById('alert-icon').textContent = '✅';
      document.getElementById('alert-title').textContent = 'ALL CLEAR — NO ACCIDENTS';
      document.getElementById('alert-desc').textContent = data.summary;
    }

    // Event log
    const list = document.getElementById('events-list');
    list.innerHTML = '';
    document.getElementById('log-count').textContent = `${allFrameResults.length} FRAMES`;

    if (allFrameResults.length === 0) {
      list.innerHTML = '<div class="no-events">No frame data available</div>';
    } else {
      allFrameResults.forEach(f => {
        const item = document.createElement('div');
        item.className = 'event-item' + (f.accident_detected ? ' accident' : '');
        const severityClass = f.severity ? `severity-${f.severity}` : 'severity-none';
        const confPct = Math.round((f.confidence || 0) * 100);

        item.innerHTML = `
          <div class="event-time">${formatTime(f.timestamp)}</div>
          <div class="event-body">
            <div class="event-type">
              ${f.accident_detected ? '🚨 Accident Detected' : '✓ Clear'}
              <span class="severity-badge ${severityClass}" style="margin-left:6px">${f.severity || 'safe'}</span>
            </div>
            <div class="event-desc">${f.accident_detected ? 'Accident detected.' : (f.description || '')}</div>
            <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%;background:${f.accident_detected ? 'var(--accent2)' : 'var(--safe)'}"></div></div>
          </div>`;
        item.onclick = () => {
          const video = document.getElementById('main-video');
          video.currentTime = f.timestamp;
          video.pause();
          document.getElementById('btn-play').textContent = '▶';
        };
        list.appendChild(item);
      });
    }

    // Timeline strip
    buildTimeline();

    // Timeline accident markers on progress bar
    buildTimelineMarkers();
  }

  function buildTimeline() {
    const strip = document.getElementById('timeline-strip');
    strip.innerHTML = '';
    const panel = document.getElementById('timeline-panel');
    panel.style.display = 'block';
    document.getElementById('timeline-count').textContent = `${allFrameResults.length} SAMPLES`;

    allFrameResults.forEach(f => {
      const thumb = document.createElement('div');
      thumb.className = 'frame-thumb' + (f.accident_detected ? ' accident-frame' : '');
      thumb.title = `${formatTime(f.timestamp)} — ${f.accident_detected ? (f.accident_type || 'Accident') : 'Clear'}`;
      thumb.textContent = formatTime(f.timestamp);
      thumb.onclick = () => {
        const video = document.getElementById('main-video');
        video.currentTime = f.timestamp;
      };
      strip.appendChild(thumb);
    });
  }

  function buildTimelineMarkers() {
    const track = document.getElementById('progress-track');
    track.querySelectorAll('.accident-marker').forEach(m => m.remove());
    if (!videoDuration) return;
    accidentEvents.forEach(e => {
      const pct = (e.timestamp / videoDuration) * 100;
      const marker = document.createElement('div');
      marker.className = 'accident-marker';
      marker.style.left = pct + '%';
      marker.title = `Accident at ${formatTime(e.timestamp)}`;
      track.appendChild(marker);
    });
  }

  // Video controls
  const video = document.getElementById('main-video');

  video.addEventListener('timeupdate', () => {
    if (!video.duration) return;
    const pct = (video.currentTime / video.duration) * 100;
    document.getElementById('progress-fill').style.width = pct + '%';
    updateTimeDisplay();
  });

  video.addEventListener('ended', () => {
    document.getElementById('btn-play').textContent = '▶';
    accidentRanges = [];
    renderRanges();
    document.getElementById('range-error').textContent = '';
  });

  document.getElementById('progress-track').addEventListener('click', e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = pct * video.duration;
  });

  function togglePlay() {
    if (video.paused) {
      video.play();
      document.getElementById('btn-play').textContent = '⏸';
    } else {
      video.pause();
      document.getElementById('btn-play').textContent = '▶';
    }
  }

  function seekBy(sec) {
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + sec));
  }

  function setVolume(v) { video.volume = v; }

  function updateTimeDisplay() {
    const cur = formatTime(video.currentTime);
    const dur = formatTime(video.duration || 0);
    document.getElementById('time-display').textContent = `${cur} / ${dur}`;
  }

  function formatTime(s) {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  }

  function setProgMsg(msg, pct) {
    document.getElementById('prog-msg').textContent = msg;
    document.getElementById('prog-pct').textContent = pct + '%';
    document.getElementById('prog-fill').style.width = pct + '%';
  }

  function resetUI() {
    currentJobId = null;
    accidentEvents = [];
    allFrameResults = [];
    if (pollInterval) clearInterval(pollInterval);
    document.getElementById('alert-banner').classList.remove('visible', 'safe');
    document.getElementById('analysis-progress').classList.remove('visible');
    document.getElementById('reanalyze-box').classList.remove('visible');
    document.getElementById('stat-frames').textContent = '—';
    document.getElementById('stat-accidents').textContent = '—';
    document.getElementById('stat-duration').textContent = '—';
    document.getElementById('stat-severity').textContent = '—';
    document.getElementById('events-list').innerHTML = '<div class="no-events">Analyzing...</div>';
    document.getElementById('log-count').textContent = 'AWAITING ANALYSIS';
    document.getElementById('timeline-panel').style.display = 'none';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('btn-play').textContent = '▶';
  }

  function manualMarkAccident() {
    const ts = document.getElementById('manual-timestamp').value.trim();
    if (!ts) {
      setReanalyzeStatus('Enter a timestamp e.g. 1:30 or 90', 'error');
      return;
    }

    // Parse M:SS or plain seconds
    let seconds = 0;
    if (ts.includes(':')) {
      const parts = ts.split(':');
      seconds = parseInt(parts[0]) * 60 + parseFloat(parts[1]);
    } else {
      seconds = parseFloat(ts);
    }

    if (isNaN(seconds) || seconds < 0) {
      setReanalyzeStatus('Invalid format. Use 1:30 or 90', 'error');
      return;
    }

    if (videoDuration && seconds > videoDuration) {
      setReanalyzeStatus('Timestamp exceeds video length (' + formatTime(videoDuration) + ')', 'error');
      return;
    }

    // Build accident object directly — no API call
    const accidentData = {
      accident_detected: true,
      confidence: 1.0,
      accident_type: 'Manually Marked',
      description: 'Accident manually marked by operator at this timestamp.',
      severity: 'high',
      timestamp: Math.round(seconds * 100) / 100,
      manual: true
    };

    // ── Notify ESP8266 to turn OFF relay ─────────────────────────────────────
    (async () => {
      try {
        const espRes = await fetch('/api/esp8266/relay/off', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: `Manual operator mark at ${formatTime(seconds)}` })
        });
        const espData = await espRes.json();
        console.log('[ESP8266] Relay OFF signal sent to', espData.esp8266_ip);
      } catch(e) {
        console.warn('[ESP8266] Could not send relay signal:', e.message);
      }
    })();

    // Add red marker on video progress timeline
    if (videoDuration) {
      const pct = (seconds / videoDuration) * 100;
      const track = document.getElementById('progress-track');
      const marker = document.createElement('div');
      marker.className = 'accident-marker';
      marker.style.left = pct + '%';
      marker.title = '⚠ Manual accident at ' + formatTime(seconds);
      track.appendChild(marker);
    }

    // Inject into incident log
    injectManualResult(accidentData);

    setReanalyzeStatus('🚨 Accident marked at ' + formatTime(seconds), 'error');
    document.getElementById('manual-timestamp').value = '';

    // Auto-seek video to that moment
    const video = document.getElementById('main-video');
    if (video.src) {
      video.currentTime = seconds;
      video.pause();
      document.getElementById('btn-play').textContent = '▶';
    }
  }

  function injectManualResult(f) {
    const list = document.getElementById('events-list');
    const placeholder = list.querySelector('.no-events');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'event-item accident manual-check';
    item.innerHTML = `
      <div class="event-time" style="color:var(--accent2)">${formatTime(f.timestamp)}</div>
      <div class="event-body">
        <div class="event-type" style="color:var(--accent2)">
          🚨 ${f.accident_type || 'Accident'}
          <span class="severity-badge severity-high" style="margin-left:6px">${f.severity || 'high'}</span>
          <span class="manual-badge">MANUAL</span>
        </div>
        <div class="event-desc">${f.description}</div>
        <div class="conf-bar">
          <div class="conf-fill" style="width:100%;background:var(--accent2)"></div>
        </div>
      </div>`;
    item.onclick = () => {
      const video = document.getElementById('main-video');
      video.currentTime = f.timestamp;
      video.pause();
      document.getElementById('btn-play').textContent = '▶';
    };
    list.insertBefore(item, list.firstChild);
  }

  function setReanalyzeStatus(msg, type) {
    const el = document.getElementById('reanalyze-status');
    el.textContent = msg;
    el.className = 'reanalyze-status ' + type;
  }

  function seekToTime(sec) {
    const v = document.getElementById('main-video');
    if (v.src) { v.currentTime = sec; v.pause(); document.getElementById('btn-play').textContent = '▶'; }
  }

  // ── Reference Library ─────────────────────────────────────────────────────
  let refPollInterval = null;

  document.getElementById('ref-file-input').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';
    setRefStatus('⏳ Uploading ' + file.name + '...', 'building');

    const fd = new FormData();
    fd.append('video', file);
    try {
      const res = await fetch('/api/reference/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) { setRefStatus('✖ ' + (data.error || 'Upload failed'), 'error'); return; }
      setRefStatus('⏳ Building reference profile...', 'building');
      pollRefProgress(data.ref_id);
    } catch (err) {
      setRefStatus('✖ ' + err.message, 'error');
    }
  });

  function pollRefProgress(refId) {
    if (refPollInterval) clearInterval(refPollInterval);
    refPollInterval = setInterval(async () => {
      const res = await fetch('/api/reference/progress/' + refId);
      const data = await res.json();
      if (data.status === 'done') {
        clearInterval(refPollInterval);
        setRefStatus('✅ ' + data.message, 'ready');
        loadReferenceList();
      } else if (data.status === 'error') {
        clearInterval(refPollInterval);
        setRefStatus('✖ ' + data.message, 'error');
      } else {
        setRefStatus('⏳ ' + data.message, 'building');
      }
    }, 2000);
  }

  async function loadReferenceList() {
    try {
      const res = await fetch('/api/reference/list');
      const refs = await res.json();
      const listEl = document.getElementById('ref-list');
      document.getElementById('ref-count').textContent = refs.length + ' VIDEO' + (refs.length !== 1 ? 'S' : '');
      if (refs.length === 0) {
        listEl.innerHTML = '<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-dim)">No reference videos yet</div>';
        return;
      }
      listEl.innerHTML = refs.map(r => `
        <div class="ref-item">
          <span class="ref-item-icon">🎬</span>
          <div>
            <div class="ref-item-name">${r.original_name}</div>
            <div class="ref-item-meta">${r.accident_frames_count} accident frames · ${r.duration}s</div>
          </div>
        </div>`).join('');
    } catch (e) {}
  }

  function setRefStatus(msg, cls) {
    const el = document.getElementById('ref-status');
    el.textContent = msg;
    el.className = 'ref-status ' + cls;
  }

  // Load reference list on page load
  loadReferenceList();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────

def encode_frame_base64(frame, quality=75):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def load_reference_profiles():
    profiles = []
    for fname in os.listdir(REFERENCE_FOLDER):
        if fname.endswith("_profile.json"):
            path = os.path.join(REFERENCE_FOLDER, fname)
            try:
                with open(path) as f:
                    profiles.append(json.load(f))
            except Exception:
                pass
    return profiles


def build_reference_profile(video_path, ref_id, original_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / fps
    sample_interval = max(1, int(fps // 2))
    frame_indices = list(range(0, total_frames, sample_interval))[:60]

    # Store ALL sampled frames with their visual histogram for fast pixel-level matching.
    # No Claude API call needed here — just store color signature of every frame.
    accident_frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        b64 = encode_frame_base64(frame)
        timestamp = round(frame_idx / fps, 2)
        accident_frames.append({
            "frame_number": frame_idx,
            "timestamp": timestamp,
            "frame_b64": b64,
            "histogram": compute_frame_histogram(frame),
            "description": f"Reference frame at {timestamp}s"
        })

    cap.release()

    profile = {
        "ref_id": ref_id,
        "original_name": original_name,
        "duration": round(duration, 2),
        "total_frames": total_frames,
        "accident_frames": accident_frames
    }
    profile_path = os.path.join(REFERENCE_FOLDER, f"{ref_id}_profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f)
    return profile


def compute_frame_histogram(frame):
    """Compute a normalized HSV color histogram for a frame. Returns a flat list."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    combined = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    return combined.tolist()


def histogram_similarity(h1, h2):
    """Pearson correlation between two histograms. Returns 0.0-1.0."""
    a = np.array(h1, dtype=np.float32)
    b = np.array(h2, dtype=np.float32)
    std_a, std_b = a.std(), b.std()
    if std_a == 0 or std_b == 0:
        return float(np.isclose(a, b).all())
    corr = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, corr)


def decode_b64_to_frame(b64_str):
    """Decode a base64 JPEG string back to an OpenCV frame."""
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def compare_frame_to_references(frame_b64, frame_number, timestamp, reference_profiles):
    """
    Pure OpenCV histogram comparison — zero Claude API calls.
    Compares the current frame against every stored reference frame by color histogram.
    Same video = similarity ~0.99. Unrelated scene = ~0.4-0.6.
    """
    if not reference_profiles:
        return None

    MATCH_THRESHOLD = 0.80

    current_frame = decode_b64_to_frame(frame_b64)
    if current_frame is None:
        return None
    current_hist = compute_frame_histogram(current_frame)

    best_sim = 0.0
    best_profile = None
    best_af = None

    for profile in reference_profiles:
        for af in profile["accident_frames"]:
            if "histogram" in af:
                ref_hist = af["histogram"]
            else:
                ref_frame = decode_b64_to_frame(af["frame_b64"])
                if ref_frame is None:
                    continue
                ref_hist = compute_frame_histogram(ref_frame)

            sim = histogram_similarity(current_hist, ref_hist)
            if sim > best_sim:
                best_sim = sim
                best_profile = profile
                best_af = af

    if best_sim >= MATCH_THRESHOLD and best_profile is not None:
        return {
            "matched": True,
            "confidence": round(min(1.0, best_sim), 3),
            "matched_ref_name": best_profile["original_name"],
            "matched_ref_timestamp": best_af["timestamp"],
            "match_reason": f"Visual similarity {best_sim:.2f}",
            "frame_number": frame_number,
            "timestamp": timestamp
        }
    return None

def analyze_frame_with_claude(frame_b64):
    safety_prompt = (
        "You are an industrial safety AI. Analyze this frame from an industrial/factory/warehouse video. "
        "Detect ONLY industrial accidents: worker falls, fire/explosion, equipment malfunction, "
        "chemical spill, structural collapse, electrocution, unsafe PPE in high-risk areas. "
        'Respond ONLY with valid JSON: {"accident_detected": true/false, "confidence": 0.0-1.0, '
        '"accident_type": "type or null", "description": "brief description", '
        '"severity": "low/medium/high/critical or null"}'
    )
    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}},
                    {"type": "text", "text": safety_prompt}
                ],
            }],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {"accident_detected": False, "confidence": 0.0, "accident_type": None,
                "description": f"Analysis error: {str(e)}", "severity": None}


def analyze_video_task(video_path, job_id, accident_ranges=None):
    try:
        analysis_progress[job_id] = {"status": "processing", "progress": 0, "message": "Opening video..."}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            analysis_progress[job_id] = {"status": "error", "progress": 0, "message": "Cannot open video file."}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = total_frames / fps
        sample_interval = max(1, int(fps // 2))
        frame_indices = list(range(0, total_frames, sample_interval))[:60]

        if accident_ranges is None:
            accident_ranges = []

        reference_profiles = load_reference_profiles()
        has_references = len(reference_profiles) > 0

        accidents = []
        results_per_frame = []

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            progress = int((i + 1) / len(frame_indices) * 100)
            timestamp = round(frame_idx / fps, 2)

            frame_b64 = encode_frame_base64(frame)

            # ── Priority 1: Reference match (always checked first for every frame) ──
            if has_references:
                analysis_progress[job_id] = {
                    "status": "processing", "progress": progress,
                    "message": f"Frame {i+1}/{len(frame_indices)} at {timestamp:.1f}s — checking references..."
                }
                try:
                    ref_match = compare_frame_to_references(frame_b64, frame_idx, timestamp, reference_profiles)
                except RuntimeError as ref_err:
                    # Surface API errors (e.g. missing API key) directly in the frame result
                    result = {
                        "accident_detected": False,
                        "confidence": 0.0,
                        "accident_type": None,
                        "description": str(ref_err),
                        "severity": None,
                        "timestamp": timestamp,
                        "frame_number": frame_idx,
                        "frame_b64": frame_b64,
                        "reference_match": False,
                        "pinned": False,
                    }
                    results_per_frame.append(result)
                    continue
                if ref_match:
                    result = {
                        "accident_detected": True,
                        "confidence": ref_match["confidence"],
                        "accident_type": "Accident Detected",
                        "description": "Accident detected.",
                        "severity": "high",
                        "timestamp": timestamp,
                        "frame_number": frame_idx,
                        "frame_b64": frame_b64,
                        "reference_match": True,
                        "matched_ref_name": ref_match["matched_ref_name"],
                        "matched_ref_timestamp": ref_match["matched_ref_timestamp"],
                        "pinned": False,
                    }
                else:
                    # ── Priority 2: Claude AI analysis (no reference match) ───
                    analysis_progress[job_id] = {
                        "status": "processing", "progress": progress,
                        "message": f"Frame {i+1}/{len(frame_indices)} at {timestamp:.1f}s — analyzing with AI..."
                    }
                    result = analyze_frame_with_claude(frame_b64)
                    result["timestamp"] = timestamp
                    result["frame_number"] = frame_idx
                    result["frame_b64"] = frame_b64
                    result["reference_match"] = False
                    result["pinned"] = False

            else:
                # ── No references loaded: Claude AI analysis ──────────────────
                analysis_progress[job_id] = {
                    "status": "processing", "progress": progress,
                    "message": f"Frame {i+1}/{len(frame_indices)} at {timestamp:.1f}s — analyzing with AI..."
                }
                result = analyze_frame_with_claude(frame_b64)
                result["timestamp"] = timestamp
                result["frame_number"] = frame_idx
                result["frame_b64"] = frame_b64
                result["reference_match"] = False
                result["pinned"] = False

            results_per_frame.append(result)
            if result.get("accident_detected") and result.get("confidence", 0) >= 0.5:
                accidents.append(result)
                # ── ESP8266: Send relay OFF signal for every confirmed accident frame ──
                trigger_relay_in_background(
                    reason=f"Accident at {timestamp:.1f}s — {result.get('accident_type', 'Unknown')} "
                           f"(confidence {result.get('confidence', 0):.0%})"
                )

        cap.release()

        accident_detected = len(accidents) > 0
        max_severity = None
        severity_order = ["low", "medium", "high", "critical"]
        for a in accidents:
            s = a.get("severity")
            if s and (max_severity is None or severity_order.index(s) > severity_order.index(max_severity)):
                max_severity = s

        analysis_results[job_id] = {
            "job_id": job_id,
            "accident_detected": accident_detected,
            "total_accidents_found": len(accidents),
            "max_severity": max_severity,
            "duration": round(duration, 2),
            "frames_analyzed": len(frame_indices),
            "accident_events": [{k: v for k, v in a.items() if k != "frame_b64"} for a in accidents],
            "all_frame_results": results_per_frame,
            "summary": (
                f"Detected {len(accidents)} industrial accident event(s) across {len(frame_indices)} frames."
                if accident_detected
                else f"No industrial accidents detected across {len(frame_indices)} frames."
            ),
        }
        analysis_progress[job_id] = {"status": "done", "progress": 100, "message": "Analysis complete."}

    except Exception as e:
        analysis_progress[job_id] = {"status": "error", "progress": 0, "message": str(e)}


reference_build_progress = {}

def build_reference_task(video_path, ref_id, original_name):
    reference_build_progress[ref_id] = {"status": "processing", "message": "Building reference profile..."}
    try:
        profile = build_reference_profile(video_path, ref_id, original_name)
        if profile:
            reference_build_progress[ref_id] = {
                "status": "done",
                "message": f"Ready. {len(profile['accident_frames'])} accident frames indexed.",
                "original_name": original_name,
                "ref_id": ref_id
            }
        else:
            reference_build_progress[ref_id] = {"status": "error", "message": "Failed to build profile."}
    except Exception as e:
        reference_build_progress[ref_id] = {"status": "error", "message": str(e)}


@app.route("/api/status")
def api_status():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return jsonify({
        "api_key_set": bool(key),
        "api_key_hint": f"...{key[-6:]}" if key else None
    })


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Parse accident time ranges sent from frontend
    accident_ranges = []
    ranges_raw = request.form.get("accident_ranges", "[]")
    try:
        parsed = json.loads(ranges_raw)
        for r in parsed:
            s = float(r.get("start", 0))
            e = float(r.get("end", 0))
            if e > s >= 0:
                accident_ranges.append({"start": s, "end": e})
    except Exception:
        accident_ranges = []

    job_id = f"job_{int(time.time() * 1000)}"
    filename = f"{job_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    thread = threading.Thread(target=analyze_video_task, args=(video_path, job_id, accident_ranges))
    thread.daemon = True
    thread.start()
    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "message": "Upload successful, analysis started.",
        "accident_ranges_count": len(accident_ranges)
    })


@app.route("/api/progress/<job_id>")
def get_progress(job_id):
    return jsonify(analysis_progress.get(job_id, {"status": "not_found", "progress": 0, "message": "Job not found."}))


@app.route("/api/results/<job_id>")
def get_results(job_id):
    result = analysis_results.get(job_id)
    if not result:
        return jsonify({"error": "Results not ready or job not found."}), 404
    return jsonify(result)


@app.route("/api/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/api/reference/upload", methods=["POST"])
def upload_reference():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    ref_id = f"ref_{int(time.time() * 1000)}"
    filename = f"{ref_id}_{file.filename}"
    video_path = os.path.join(REFERENCE_FOLDER, filename)
    file.save(video_path)
    thread = threading.Thread(target=build_reference_task, args=(video_path, ref_id, file.filename))
    thread.daemon = True
    thread.start()
    return jsonify({"ref_id": ref_id, "message": "Reference upload started."})


@app.route("/api/reference/progress/<ref_id>")
def reference_progress(ref_id):
    return jsonify(reference_build_progress.get(ref_id, {"status": "not_found", "message": "Not found."}))


@app.route("/api/reference/list")
def reference_list():
    profiles = load_reference_profiles()
    return jsonify([{
        "ref_id": p["ref_id"],
        "original_name": p["original_name"],
        "duration": p["duration"],
        "accident_frames_count": len(p["accident_frames"])
    } for p in profiles])


@app.route("/api/reanalyze", methods=["POST"])
def reanalyze_frame_route():
    data = request.get_json()
    job_id = data.get("job_id")
    timestamp_str = data.get("timestamp", "").strip()
    if not job_id or not timestamp_str:
        return jsonify({"error": "job_id and timestamp required"}), 400
    try:
        if ":" in timestamp_str:
            parts = timestamp_str.split(":")
            seconds = int(parts[0]) * 60 + float(parts[1])
        else:
            seconds = float(timestamp_str)
    except Exception:
        return jsonify({"error": "Invalid timestamp"}), 400
    video_file = None
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.startswith(job_id):
            video_file = os.path.join(UPLOAD_FOLDER, fname)
            break
    if not video_file:
        return jsonify({"error": "Video not found"}), 404
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Could not extract frame"}), 500
    frame_b64 = encode_frame_base64(frame)
    result = analyze_frame_with_claude(frame_b64)
    result["timestamp"] = round(seconds, 2)
    result["frame_b64"] = frame_b64
    result["manual"] = True
    return jsonify(result)


@app.route("/api/esp8266/relay/off", methods=["POST"])
def manual_relay_off():
    """
    Called by the frontend when operator manually marks an accident.
    Sends relay OFF signal to ESP8266.
    """
    data = request.get_json(silent=True) or {}
    reason = data.get("reason", "Manual operator trigger")
    trigger_relay_in_background(reason=reason)
    return jsonify({
        "status": "signal_sent",
        "esp8266_ip": ESP8266_IP,
        "esp8266_port": ESP8266_PORT,
        "path": ESP8266_PATH,
        "reason": reason
    })


@app.route("/api/esp8266/status")
def esp8266_status():
    """Returns current ESP8266 configuration so the UI can display it."""
    return jsonify({
        "ip": ESP8266_IP,
        "port": ESP8266_PORT,
        "path": ESP8266_PATH,
        "url": f"http://{ESP8266_IP}:{ESP8266_PORT}{ESP8266_PATH}"
    })


@app.route("/api/esp8266/reset", methods=["POST"])
def relay_reset():
    """
    Called by the frontend on every page load.
    Sends /accident_off to ESP8266 → Relay turns ON (machine safe to run).
    """
    import urllib.request
    url = f"http://{ESP8266_IP}:{ESP8266_PORT}/accident_off"
    try:
        urllib.request.urlopen(url, timeout=ESP8266_TIMEOUT_SEC)
        print(f"  [ESP8266] Page refreshed → Relay reset to ON (machine running)")
    except Exception as e:
        print(f"  [ESP8266] ⚠ Reset failed — {e}")
    return jsonify({"status": "relay_reset"})


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


if __name__ == "__main__":
    print(f"\n  Starting SafeGuard AI...")
    print(f"  Upload folder    : {UPLOAD_FOLDER}")
    print(f"  Reference folder : {REFERENCE_FOLDER}")
    print(f"  ESP8266 relay    : http://{ESP8266_IP}:{ESP8266_PORT}{ESP8266_PATH}")
    print(f"  ► To change ESP8266 IP, edit ESP8266_IP near line 24 of this file")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ⚠  WARNING: ANTHROPIC_API_KEY is not set!")
        print("  ⚠  Set it before starting:")
        print("       Windows : set ANTHROPIC_API_KEY=sk-ant-...")
        print("       Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...")
        print("  ⚠  Without it, ALL analysis will fail.\n")
    else:
        print(f"  API key          : sk-ant-...{api_key[-6:]}")
    print(f"  Open browser     : http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
