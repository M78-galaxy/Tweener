import base64
import json
import urllib.request
from pathlib import Path

from krita import Extension, Krita
from PyQt5.QtWidgets import QMessageBox

BASE_URL = "http://127.0.0.1:8766"
HEALTH_URL = BASE_URL + "/health"
INTERPOLATE_URL = BASE_URL + "/v1/stickman/interpolate"
REQUEST_PATH = Path.home() / "Desktop" / "tweener_request.json"
OUTPUT_PNG = Path.home() / "Desktop" / "tweener_krita_prediction.png"


class TweenerHealthExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        health = window.createAction(
            "tweener_health_check",
            "Tweener: Check Server",
            "tools/scripts",
        )
        health.triggered.connect(self.check_health)

        demo = window.createAction(
            "tweener_interpolate_demo",
            "Tweener: Interpolate Demo",
            "tools/scripts",
        )
        demo.triggered.connect(self.interpolate_demo)

    def check_health(self):
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = "\n".join(
                [
                    "status: " + str(data.get("status")),
                    "device: " + str(data.get("device")),
                    "epoch: " + str(data.get("checkpoint_epoch")),
                    "context: " + str(data.get("context")),
                    "strides: " + str(data.get("strides")),
                    "url: " + HEALTH_URL,
                ]
            )
            QMessageBox.information(None, "Tweener Health", text)
        except Exception as exc:
            QMessageBox.critical(
                None,
                "Tweener Health",
                "request failed: " + str(exc) + "\nurl: " + HEALTH_URL,
            )

    def _place_png(self, app):
        doc = app.activeDocument()
        if doc is None:
            doc = app.openDocument(str(OUTPUT_PNG))
            if doc is None:
                raise RuntimeError("openDocument failed")
            app.activeWindow().addView(doc)
            doc.waitForDone()
            return "opened new document"

        layer_name = "Tweener Prediction"
        layer = doc.createFileLayer(
            layer_name,
            str(OUTPUT_PNG),
            "None",
        )
        if layer is None:
            raise RuntimeError("createFileLayer failed")
        doc.rootNode().addChildNode(layer, None)
        doc.refreshProjection()
        doc.waitForDone()
        return "added layer: " + layer_name

    def interpolate_demo(self):
        try:
            if not REQUEST_PATH.is_file():
                raise FileNotFoundError(
                    "missing request file: " + str(REQUEST_PATH)
                )

            request_data = json.loads(
                REQUEST_PATH.read_text(encoding="utf-8")
            )
            body = json.dumps(request_data).encode("utf-8")
            req = urllib.request.Request(
                INTERPOLATE_URL,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            png = base64.b64decode(data["png_base64"])
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("invalid png bytes")

            OUTPUT_PNG.write_bytes(png)
            placed = self._place_png(Krita.instance())

            text = "\n".join(
                [
                    "request_id: " + str(data.get("request_id")),
                    "stride: " + str(data.get("stride")),
                    "device: " + str(data.get("device")),
                    "epoch: " + str(data.get("checkpoint_epoch")),
                    "png_bytes: " + str(len(png)),
                    "result: " + placed,
                ]
            )
            QMessageBox.information(None, "Tweener Interpolate", text)
        except Exception as exc:
            QMessageBox.critical(
                None,
                "Tweener Interpolate",
                "interpolate failed: " + str(exc),
            )


Krita.instance().addExtension(TweenerHealthExtension(Krita.instance()))
