import os
import time
import asyncio
from typing import Optional, Dict, Any
import httpx

FIREBASE_API_KEY = "AIzaSyCYuXqbJ0YBNltoGS4-7Y6Hozrra8KKmaE"
FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
FIREBASE_REFRESH_URL = f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}"

def _get_proxy() -> Optional[str]:
    """Get proxy URL from environment."""
    return os.getenv("HTTP_PROXY", "").strip() or None

def _get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Create httpx client with optional proxy support."""
    proxy = _get_proxy()
    if proxy:
        mounts = {
            "https://": httpx.AsyncHTTPTransport(proxy=proxy),
            "http://": httpx.AsyncHTTPTransport(proxy=proxy),
        }
        return httpx.AsyncClient(mounts=mounts, timeout=timeout)
    return httpx.AsyncClient(timeout=timeout)

class GumloopAuth:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.id_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.expires_at: float = 0
        self._lock = asyncio.Lock()

    async def login(self, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        payload = {
            "returnSecureToken": True,
            "email": self.email,
            "password": self.password,
            "clientType": "CLIENT_TYPE_WEB"
        }
        close_client = client is None
        if client is None:
            client = _get_http_client()
        try:
            resp = await client.post(FIREBASE_AUTH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.id_token = data.get("idToken")
            self.refresh_token = data.get("refreshToken")
            self.user_id = data.get("localId")
            expires_in = int(data.get("expiresIn", 3600))
            self.expires_at = time.time() + expires_in - 300
            return data
        finally:
            if close_client:
                await client.aclose()

    async def refresh(self, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        if not self.refresh_token:
            return await self.login(client)
        payload = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        close_client = client is None
        if client is None:
            client = _get_http_client()
        try:
            resp = await client.post(FIREBASE_REFRESH_URL, data=payload)
            resp.raise_for_status()
            data = resp.json()
            self.id_token = data.get("id_token")
            self.refresh_token = data.get("refresh_token", self.refresh_token)
            self.user_id = data.get("user_id", self.user_id)
            expires_in = int(data.get("expires_in", 3600))
            self.expires_at = time.time() + expires_in - 300
            return data
        except httpx.HTTPError:
            return await self.login(client)
        finally:
            if close_client:
                await client.aclose()

    async def get_token(self, client: Optional[httpx.AsyncClient] = None) -> str:
        async with self._lock:
            if not self.id_token or time.time() >= self.expires_at:
                if self.refresh_token:
                    await self.refresh(client)
                else:
                    await self.login(client)
            return self.id_token

_auth_instance: Optional[GumloopAuth] = None

def get_auth() -> GumloopAuth:
    global _auth_instance
    if _auth_instance is None:
        email = os.getenv("GUMLOOP_EMAIL", "")
        password = os.getenv("GUMLOOP_PASSWORD", "")
        _auth_instance = GumloopAuth(email, password)
    return _auth_instance
