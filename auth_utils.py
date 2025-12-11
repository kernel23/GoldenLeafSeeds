import hashlib
import os
import hmac

def hash_password(password):
    """Hashes a password with a fixed salt (for this demo)."""
    # In a real app, use a per-user salt stored in DB.
    salt = "somesalt"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(stored_hash, password):
    """Verifies a password against a stored hash using constant-time comparison."""
    return hmac.compare_digest(stored_hash, hash_password(password))
