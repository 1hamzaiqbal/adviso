import os
import requests
from typing import Optional, Tuple, Any, Dict


class CloudBrandAnalyzer:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url or os.getenv("ADVALUATE_BACKEND_URL", "").strip()
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def _sign_upload(self, filename: str, content_type: str) -> Tuple[str, str]:
        url = f"{self.base_url}/api/sign-upload"
        r = requests.get(url, params={"filename": filename, "contentType": content_type}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["uploadUrl"], data["gcsUri"]

    def _upload_bytes(self, upload_url: str, data: bytes, content_type: str) -> None:
        headers = {"Content-Type": content_type or "video/mp4"}
        r = requests.put(upload_url, data=data, headers=headers, timeout=self.timeout)
        r.raise_for_status()

    def _analyze(self, gcs_uri: str, brand_name: Optional[str], brand_mission: Optional[str]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/analyze"
        payload = {
            "videoGcsUri": gcs_uri,
            "brandName": brand_name or None,
            "brandMission": brand_mission or None,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        if r.ok:
            return r.json()
        # Try to surface helpful server details
        detail = None
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        msg = f"Analyze request failed ({r.status_code}). Details: {detail}"
        raise RuntimeError(msg)

    def run(self,
            video_bytes: bytes,
            filename: str = "video.mp4",
            content_type: str = "video/mp4",
            brand_name: Optional[str] = None,
            brand_mission: Optional[str] = None) -> Dict[str, Any]:
        """Uploads the video to the backend using signed URL and triggers analysis.

        Returns the JSON result produced by the backend.
        """
        if not self.is_configured():
            raise RuntimeError("CloudBrandAnalyzer not configured. Set ADVALUATE_BACKEND_URL or pass base_url.")

        upload_url, gcs_uri = self._sign_upload(filename, content_type)
        self._upload_bytes(upload_url, video_bytes, content_type)
        return self._analyze(gcs_uri, brand_name, brand_mission)
