"""SQLite database storage for biometric templates.

This module provides persistent storage for enrolled users and their
biometric templates (face encodings, and later iris codes).
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from ..utils.config import get_config


@dataclass
class UserRecord:
    """Represents an enrolled user."""
    
    user_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class FaceTemplateRecord:
    """Represents a stored face template."""
    
    template_id: int
    user_id: str
    encodings: list[np.ndarray]  # List of 128-D face encodings
    created_at: datetime
    quality_scores: list[float] = field(default_factory=list)
    
    def get_average_encoding(self) -> np.ndarray:
        """Get the average encoding for matching."""
        if not self.encodings:
            raise ValueError("No encodings available")
        return np.mean(self.encodings, axis=0)


@dataclass
class EnrollmentResult:
    """Result of an enrollment operation."""
    
    success: bool
    user_id: Optional[str] = None
    template_id: Optional[int] = None
    message: str = ""
    num_samples: int = 0


class DatabaseManager:
    """Manages SQLite database for biometric storage.
    
    Thread-safe database operations for storing and retrieving
    user records and biometric templates.
    
    Usage:
        db = DatabaseManager()
        db.initialize()
        
        # Enroll a new user
        result = db.enroll_user("user123", "John Doe", face_encodings)
        
        # Get user's face template
        template = db.get_face_template("user123")
        
        # List all enrolled users
        users = db.list_users()
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file.
                     If None, uses path from config.
        """
        if db_path is None:
            config = get_config()
            db_path = config.database.path
        
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.RLock()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def initialize(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Face templates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_templates (
                template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                encodings BLOB NOT NULL,
                quality_scores TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
                    ON DELETE CASCADE
            )
        """)
        
        # Iris templates table (for future use)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iris_templates (
                template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                iris_code BLOB NOT NULL,
                mask BLOB,
                eye TEXT CHECK(eye IN ('left', 'right')),
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
                    ON DELETE CASCADE
            )
        """)
        
        # Verification logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                verification_type TEXT,
                success INTEGER,
                face_score REAL,
                iris_score REAL,
                combined_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_templates_user_id 
            ON face_templates(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iris_templates_user_id 
            ON iris_templates(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_verification_logs_user_id 
            ON verification_logs(user_id)
        """)
        
        conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    # -------------------- User Operations --------------------
    
    def add_user(
        self,
        user_id: str,
        name: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """Add a new user to the database.
        
        Args:
            user_id: Unique user identifier
            name: User's display name
            metadata: Optional metadata dictionary
            
        Returns:
            True if user was added successfully
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, name, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, name, json.dumps(metadata or {}))
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # User already exists
                return False
    
    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Get user record by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserRecord if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return UserRecord(
            user_id=row['user_id'],
            name=row['name'],
            created_at=datetime.fromisoformat(row['created_at']) if isinstance(row['created_at'], str) else row['created_at'],
            updated_at=datetime.fromisoformat(row['updated_at']) if isinstance(row['updated_at'], str) else row['updated_at'],
            is_active=bool(row['is_active']),
            metadata=json.loads(row['metadata'])
        )
    
    def list_users(self, active_only: bool = True) -> list[UserRecord]:
        """List all enrolled users.
        
        Args:
            active_only: If True, only return active users
            
        Returns:
            List of UserRecord objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute(
                "SELECT * FROM users WHERE is_active = 1 ORDER BY name"
            )
        else:
            cursor.execute("SELECT * FROM users ORDER BY name")
        
        users = []
        for row in cursor.fetchall():
            users.append(UserRecord(
                user_id=row['user_id'],
                name=row['name'],
                created_at=datetime.fromisoformat(row['created_at']) if isinstance(row['created_at'], str) else row['created_at'],
                updated_at=datetime.fromisoformat(row['updated_at']) if isinstance(row['updated_at'], str) else row['updated_at'],
                is_active=bool(row['is_active']),
                metadata=json.loads(row['metadata'])
            ))
        
        return users
    
    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """Update user record.
        
        Args:
            user_id: User identifier
            name: New name (optional)
            is_active: New active status (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if user was updated
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            updates = ["updated_at = CURRENT_TIMESTAMP"]
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(int(is_active))
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            params.append(user_id)
            
            cursor.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?",
                params
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and all associated templates.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user was deleted
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    # -------------------- Face Template Operations --------------------
    
    def save_face_template(
        self,
        user_id: str,
        encodings: list[np.ndarray],
        quality_scores: Optional[list[float]] = None
    ) -> Optional[int]:
        """Save face template for a user.
        
        Args:
            user_id: User identifier
            encodings: List of face encodings (128-D numpy arrays)
            quality_scores: Optional quality scores for each encoding
            
        Returns:
            Template ID if saved successfully, None otherwise
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Serialize encodings as numpy array bytes
            encodings_array = np.array(encodings)
            encodings_blob = encodings_array.tobytes()
            
            # Store shape info in the blob for reconstruction
            shape_info = np.array(encodings_array.shape, dtype=np.int64)
            combined_blob = shape_info.tobytes() + encodings_blob
            
            cursor.execute(
                """
                INSERT INTO face_templates (user_id, encodings, quality_scores)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    combined_blob,
                    json.dumps(quality_scores or [])
                )
            )
            conn.commit()
            
            return cursor.lastrowid
    
    def get_face_template(self, user_id: str) -> Optional[FaceTemplateRecord]:
        """Get face template for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            FaceTemplateRecord if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT * FROM face_templates 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Deserialize encodings
        combined_blob = row['encodings']
        
        # First 16 bytes are shape (2 int64 values for 2D array)
        shape = np.frombuffer(combined_blob[:16], dtype=np.int64)
        encodings_data = np.frombuffer(combined_blob[16:], dtype=np.float64)
        encodings_array = encodings_data.reshape(shape)
        encodings = [enc for enc in encodings_array]
        
        return FaceTemplateRecord(
            template_id=row['template_id'],
            user_id=row['user_id'],
            encodings=encodings,
            created_at=datetime.fromisoformat(row['created_at']) if isinstance(row['created_at'], str) else row['created_at'],
            quality_scores=json.loads(row['quality_scores'])
        )
    
    def get_all_face_templates(self) -> list[FaceTemplateRecord]:
        """Get all face templates for identification.
        
        Returns:
            List of all face templates
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ft.* FROM face_templates ft
            INNER JOIN users u ON ft.user_id = u.user_id
            WHERE u.is_active = 1
            ORDER BY ft.user_id, ft.created_at DESC
        """)
        
        templates = []
        seen_users = set()
        
        for row in cursor.fetchall():
            # Only keep the most recent template per user
            if row['user_id'] in seen_users:
                continue
            seen_users.add(row['user_id'])
            
            # Deserialize encodings
            combined_blob = row['encodings']
            shape = np.frombuffer(combined_blob[:16], dtype=np.int64)
            encodings_data = np.frombuffer(combined_blob[16:], dtype=np.float64)
            encodings_array = encodings_data.reshape(shape)
            encodings = [enc for enc in encodings_array]
            
            templates.append(FaceTemplateRecord(
                template_id=row['template_id'],
                user_id=row['user_id'],
                encodings=encodings,
                created_at=datetime.fromisoformat(row['created_at']) if isinstance(row['created_at'], str) else row['created_at'],
                quality_scores=json.loads(row['quality_scores'])
            ))
        
        return templates
    
    def delete_face_template(self, user_id: str) -> bool:
        """Delete face template for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if template was deleted
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM face_templates WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            
            return cursor.rowcount > 0

    def save_iris_template(
            self,
            user_id: str,
            iris_code: np.ndarray,
            eye: str,
            mask: Optional[np.ndarray] = None,
            quality_score: Optional[float] = None
    ) -> Optional[int]:
        """Save iris template for a user."""

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            iris_blob = iris_code.tobytes()
            mask_blob = mask.tobytes() if mask is not None else None

            cursor.execute(
                """
                INSERT INTO iris_templates
                (user_id, iris_code, mask, eye, quality_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    iris_blob,
                    mask_blob,
                    eye,
                    quality_score
                )
            )

            conn.commit()
            return cursor.lastrowid

    def get_iris_template(
            self,
            user_id: str,
            eye: str
    ) -> Optional[np.ndarray]:
        """Get iris template for a user."""

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT iris_code FROM iris_templates
            WHERE user_id = ? AND eye = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, eye)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return np.frombuffer(row['iris_code'], dtype=np.float32)
    # -------------------- Enrollment Operations --------------------
    
    def enroll_user(
        self,
        user_id: str,
        name: str,
        face_encodings: list[np.ndarray],
        quality_scores: Optional[list[float]] = None,
        metadata: Optional[dict] = None
    ) -> EnrollmentResult:
        """Enroll a new user with face template.
        
        This is a convenience method that creates the user and
        saves their face template in a single transaction.
        
        Args:
            user_id: Unique user identifier
            name: User's display name
            face_encodings: List of face encodings
            quality_scores: Optional quality scores
            metadata: Optional user metadata
            
        Returns:
            EnrollmentResult with status and details
        """
        with self._lock:
            # Check if user already exists
            existing = self.get_user(user_id)
            if existing:
                return EnrollmentResult(
                    success=False,
                    user_id=user_id,
                    message=f"User '{user_id}' already enrolled"
                )
            
            # Validate encodings
            if not face_encodings:
                return EnrollmentResult(
                    success=False,
                    message="No face encodings provided"
                )
            
            # Add user
            if not self.add_user(user_id, name, metadata):
                return EnrollmentResult(
                    success=False,
                    message="Failed to create user record"
                )
            
            # Save face template
            template_id = self.save_face_template(
                user_id,
                face_encodings,
                quality_scores
            )
            
            if template_id is None:
                # Rollback user creation
                self.delete_user(user_id)
                return EnrollmentResult(
                    success=False,
                    message="Failed to save face template"
                )
            
            return EnrollmentResult(
                success=True,
                user_id=user_id,
                template_id=template_id,
                message=f"Successfully enrolled '{name}' with {len(face_encodings)} face samples",
                num_samples=len(face_encodings)
            )
    
    # -------------------- Logging Operations --------------------
    
    def log_verification(
        self,
        user_id: Optional[str],
        verification_type: str,
        success: bool,
        face_score: Optional[float] = None,
        iris_score: Optional[float] = None,
        combined_score: Optional[float] = None
    ) -> None:
        """Log a verification attempt.
        
        Args:
            user_id: User ID if identified, None if unknown
            verification_type: Type of verification ('face', 'iris', 'combined')
            success: Whether verification was successful
            face_score: Face matching score
            iris_score: Iris matching score  
            combined_score: Combined fusion score
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO verification_logs 
                (user_id, verification_type, success, face_score, iris_score, combined_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, verification_type, int(success), face_score, iris_score, combined_score)
            )
            conn.commit()
    
    def get_verification_stats(
        self,
        user_id: Optional[str] = None,
        days: int = 30
    ) -> dict:
        """Get verification statistics.
        
        Args:
            user_id: Filter by user ID (optional)
            days: Number of days to look back
            
        Returns:
            Dictionary with statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        base_query = """
            SELECT 
                COUNT(*) as total,
                SUM(success) as successful,
                AVG(face_score) as avg_face_score,
                AVG(iris_score) as avg_iris_score,
                AVG(combined_score) as avg_combined_score
            FROM verification_logs
            WHERE timestamp >= datetime('now', ?)
        """
        
        params = [f'-{days} days']
        
        if user_id:
            base_query += " AND user_id = ?"
            params.append(user_id)
        
        cursor.execute(base_query, params)
        row = cursor.fetchone()
        
        total = row['total'] or 0
        successful = row['successful'] or 0
        
        return {
            'total_attempts': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'avg_face_score': row['avg_face_score'],
            'avg_iris_score': row['avg_iris_score'],
            'avg_combined_score': row['avg_combined_score']
        }
