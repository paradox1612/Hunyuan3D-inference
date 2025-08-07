import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError
from ..config import settings

logger = logging.getLogger(__name__)

class R2Storage:
    """A class to handle file storage with Cloudflare R2"""
    
    def __init__(self):
        """Initialize the R2 storage client"""
        self.s3_client = None
        self.bucket_name = settings.R2_BUCKET_NAME
        self.public_url = settings.R2_PUBLIC_URL
        
        if not all([settings.R2_ACCOUNT_ID, settings.R2_ACCESS_KEY_ID, settings.R2_SECRET_ACCESS_KEY]):
            logger.warning("R2 credentials not fully configured. File uploads will be disabled.")
            return
            
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
            logger.info("Successfully initialized R2 storage client")
        except Exception as e:
            logger.error(f"Failed to initialize R2 storage client: {str(e)}")
    
    def upload_file(
        self,
        local_path: str,
        key: str,
        content_type: str = None,
        make_public: bool = True
    ) -> Optional[str]:
        """
        Upload a file to R2 storage
        
        Args:
            local_path: Path to the local file
            key: S3 object key
            content_type: Optional content type
            make_public: Whether to make the file publicly accessible
            
        Returns:
            Public URL if successful, None otherwise
        """
        if not self.s3_client or not self.bucket_name:
            logger.warning("R2 storage not configured, skipping upload")
            return None
            
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if make_public:
                extra_args['ACL'] = 'public-read'
            
            logger.info(f"Uploading {local_path} to R2 with key {key}")
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                key,
                ExtraArgs=extra_args
            )
            
            if self.public_url:
                public_url = f"{self.public_url.rstrip('/')}/{key}"
                logger.info(f"File uploaded successfully: {public_url}")
                return public_url
            else:
                logger.info("File uploaded successfully (no public URL configured)")
                return f"r2://{self.bucket_name}/{key}"
                
        except ClientError as e:
            logger.error(f"Error uploading to R2: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {str(e)}")
            return None

# Global storage instance
storage = R2Storage()

def upload_to_r2(
    local_path: str,
    key: str,
    content_type: str = None,
    make_public: bool = True
) -> Optional[str]:
    """
    Upload a file to R2 storage
    
    Args:
        local_path: Path to the local file
        key: S3 object key
        content_type: Optional content type
        make_public: Whether to make the file publicly accessible
        
    Returns:
        Public URL if successful, None otherwise
    """
    return storage.upload_file(local_path, key, content_type, make_public)

def ensure_directory_exists(directory: str) -> None:
    """Ensure a directory exists, create it if it doesn't"""
    os.makedirs(directory, exist_ok=True)
