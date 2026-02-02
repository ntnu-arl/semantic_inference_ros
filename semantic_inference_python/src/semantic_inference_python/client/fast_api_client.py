"""FastAPI client for InstructBLIP model interaction."""
from typing import List
import requests
import time
import os

import torch

from semantic_inference_python.client.config import FastAPIClientConfig


class InstructBlipFastAPIClient:
    """Client to interact with the InstructBLIP model via FastAPI."""

    def __init__(self, config: FastAPIClientConfig):
        """Initialize the client with the provided configuration.
        :param config: Client configuration containing API key, base URL, and endpoints."""
        self.config = config
        self.headers = {
            "X-API-Key": os.getenv("FASTAPI_API_KEY", ""),
            "Content-Type": "application/json",
        }
        self.submit_url = f"{self.config.base_url}/{self.config.submit_endpoint}/"
        self.result_url = f"{self.config.base_url}/{self.config.result_endpoint}/"
        self.timeout = self.config.timeout
        self.wait_interval = self.config.wait_interval

    def submit_job(self, prompts: List[str], features: torch.Tensor) -> str:
        """Submit a job to the FastAPI server.
        :param prompts: List of prompts to send to the model.
        :param features: Tensor of features extracted from the image.
        :return: Job ID returned by the server."""
        payload = {
            "deterministic": self.config.deterministic,
            "prompts": prompts,
            "features": features.tolist(),
        }
        response = requests.post(
            self.submit_url,
            json=payload,
            headers=self.headers,
            verify=self.config.verify_ssl,
        )
        if self.config.logging:
            print(f"Submitted job with job ID: {response.json().get('job_id', '')}")

        return response.json().get("job_id", "")

    def get_result(self, job_id: str) -> dict:
        """Poll the server for the result of a submitted job.
        :param job_id: The ID of the job to check.
        :return: The result of the job if completed, otherwise an empty dict."""
        url = f"{self.result_url}{job_id}"
        start_time = time.time()

        while True:
            response = requests.get(
                url, headers=self.headers, verify=self.config.verify_ssl
            )
            data = response.json()
            if data.get("status") == "completed":
                if self.config.logging:
                    print(f"Job {job_id} completed successfully.")
                return data.get("result", [])
            elif data.get("status") == "failed":
                if self.config.logging:
                    print(
                        f"Job {job_id} failed with error: {data.get('error', 'Unknown error')}"
                    )
                return {"error": data.get("error", "Unknown error")}
            else:
                if self.config.logging:
                    print(f"Job {job_id} still processing...")
            time.sleep(self.wait_interval)
            if time.time() - start_time > self.timeout:
                if self.config.logging:
                    print(f"Timeout waiting for job {job_id} to complete.")
                return {"error": "Timeout waiting for job completion."}
