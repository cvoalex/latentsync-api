"""
API Client for LatentSync Service
Handles all API communication with the server
"""

import json
import logging
import os
import requests
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class APIClient:
    """Client for interacting with the avatar generation API"""
    
    def __init__(self, endpoint: str = None, auth_token: str = None):
        self.endpoint = endpoint or os.environ.get('WEBSERVER_URL')
        self.auth_token = auth_token or os.environ.get('API_AUTH_TOKEN')
        
        if not self.endpoint:
            raise ValueError("WEBSERVER_URL not provided or set in environment")
        if not self.auth_token:
            raise ValueError("API_AUTH_TOKEN not provided or set in environment")
            
        # Ensure endpoint ends with /
        if not self.endpoint.endswith('/'):
            self.endpoint += '/'
            
    def avatar_next(self) -> Optional[Dict]:
        """
        Poll API for next generation task
        
        Returns:
            Dict with task details or None if no tasks available
        """
        try:
            url = f"{self.endpoint}api/avatar_next?auth={self.auth_token}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"avatar_next response: {data}")
            
            if "error" in data:
                # No tasks available
                return None
                
            try:
                return {
                    'userid': data['userid'],
                    'id': data['id'],
                    'videourl': data['videourl'],
                    'audiourl': data['audiourl'],
                    'mouth': data.get('mouth', True),
                    'steps': data.get('steps', 20),
                    'start_time': data.get('start_time', None)
                }
            except KeyError as e:
                logger.error(f"Missing expected key in response: {e}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return None
            
    def avatar_media(self, request_id: str, user_auth_token: str) -> Optional[str]:
        """
        Get media URL for a specific request
        
        Args:
            request_id: The request ID
            user_auth_token: User's auth token
            
        Returns:
            Media URL or None
        """
        try:
            url = f"{self.endpoint}avatar/media/{request_id}?auth={user_auth_token}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"avatar_media response: {data}")
            
            if 'result' in data and data['result']:
                first_result = data['result'][0]
                return first_result.get("url")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return None
            
    def get_affine_cache(self, video: str, user_auth_token: str) -> Optional[str]:
        """
        Get affine cache for a video
        
        Args:
            video: Video identifier
            user_auth_token: User's auth token
            
        Returns:
            Cache URL or None
        """
        data = {
            "params": {
                "userid": "squirrel",
                "auth": user_auth_token,
                "sys": "avatar",
                "act": "cache"
            },
            "values": {
                "video": video
            }
        }
        
        result = self.post_api(data, user_auth_token)
        
        if result is None:
            return None
            
        # Check different possible locations for the URL
        if 'cmd' in result and 'url' in result['cmd']:
            return result['cmd']['url']
        if 'url' in result:
            return result['url']
            
        return None
        
    def avatar_affinemap(self, video: str, affine: str) -> None:
        """
        Update affine map for a video
        
        Args:
            video: Video identifier
            affine: Affine transformation data
        """
        user_auth_token = self.auth_token
        data = {
            "params": {
                "auth": user_auth_token,
                "sys": "avatar",
                "act": "affinemap"
            },
            "values": {
                "video": video,
                "affine": affine
            }
        }
        
        self.post_api(data, user_auth_token)
        
    def generate_update(self, request_id: str, video_id: str, output_id: str, 
                       status: str, err_msg: str, user_auth_token: str = None,
                       end_position_info: Dict = None) -> None:
        """
        Update generation status via API
        
        Args:
            request_id: The request ID
            video_id: Video ID
            output_id: Output ID
            status: Status (processing, completed, failed)
            err_msg: Error message if failed
            user_auth_token: User's auth token (optional, uses default if not provided)
            end_position_info: Dictionary containing end position information
        """
        try:
            url = f"{self.endpoint}api"
            user_auth_token = user_auth_token or self.auth_token
            
            data = {
                "params": {
                    "auth": user_auth_token,
                    "sys": "avatar",
                    "act": "update"
                },
                "values": {
                    "id": request_id,
                    "videoid": video_id,
                    "outputid": output_id,
                    "status": status,
                    "errormessage": err_msg
                }
            }
            
            # Add end position info if provided
            if end_position_info and status == "completed":
                data["values"]["end_position"] = {
                    "end_time": end_position_info.get('end_time', 0.0),
                    "end_frame": end_position_info.get('end_frame', 0),
                    "loop_count": end_position_info.get('loop_count', 0),
                    "is_backward": end_position_info.get('is_backward', False),
                    "start_time": end_position_info.get('start_time', 0.0)
                }
            
            headers = {
                'Authorization': f'{user_auth_token}'
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            logger.info(f"Updated status for request {request_id}: {status}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update status: {e}")
            
    def post_api(self, data: Dict, user_auth_token: str = None) -> Optional[Dict]:
        """
        Generic POST request to API
        
        Args:
            data: Request data
            user_auth_token: User's auth token (optional, uses default if not provided)
            
        Returns:
            Response data or None
        """
        try:
            url = f"{self.endpoint}api"
            user_auth_token = user_auth_token or self.auth_token
            
            headers = {
                'Authorization': f'{user_auth_token}'
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.debug(f"POST request data: {data}")
            logger.debug(f"POST response: {response.text}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request exception: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode exception: {e}")
            return None
            
    def get_user(self, user_id: str) -> str:
        """
        Get user auth token
        
        Args:
            user_id: User ID
            
        Returns:
            User auth token
        """
        # This appears to be hardcoded in the original code
        # You may want to implement proper user lookup
        return "FF0uVQNCBLJ7KI"
        
    def download_file(self, url: str, dest_path: str) -> bool:
        """
        Download file from URL to destination path
        
        Args:
            url: Source URL
            dest_path: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            logger.info(f"Downloaded {url} to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
            
    def upload_file(self, file_path: str, upload_url: str) -> bool:
        """
        Upload file to specified URL
        
        Args:
            file_path: Path to file to upload
            upload_url: Upload destination URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(upload_url, files=files, timeout=300)
                response.raise_for_status()
                
            logger.info(f"Uploaded {file_path} to {upload_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
            
    def get_upload_url(self, request_id: str, user_id: str) -> Optional[str]:
        """
        Get upload URL for the output video
        
        Args:
            request_id: The request ID
            user_id: User ID
            
        Returns:
            Upload URL or None
        """
        try:
            data = {
                "params": {
                    "auth": self.auth_token,
                    "sys": "avatar",
                    "act": "get_upload_url"
                },
                "values": {
                    "id": request_id,
                    "userid": user_id
                }
            }
            
            result = self.post_api(data)
            if result:
                return result.get('upload_url')
            return None
            
        except Exception as e:
            logger.error(f"Failed to get upload URL: {e}")
            return None