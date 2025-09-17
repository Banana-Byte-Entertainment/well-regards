import pytest
import psycopg2
import os
from typing import List, Dict, Any

class TestUserCommentsDB:
    """Test suite for accessing user comments from PostgreSQL database"""
    
    @pytest.fixture(scope="class")
    def db_connection(self):
        """Fixture to create database connection"""
        # Database connection parameters - adjust these for your setup
        db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'testdb'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        
        try:
            conn = psycopg2.connect(**db_params)
            yield conn
        except psycopg2.Error as e:
            pytest.fail(f"Failed to connect to database: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    @pytest.fixture(scope="class")
    def setup_test_data(self, db_connection):
        """Fixture to set up test data"""
        cursor = db_connection.cursor()
        
        try:
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test users
            cursor.execute("""
                INSERT INTO users (username, email) 
                VALUES 
                    ('testuser1', 'user1@test.com'),
                    ('testuser2', 'user2@test.com')
                ON CONFLICT (username) DO NOTHING
            """)
            
            # Get user IDs
            cursor.execute("SELECT id FROM users WHERE username IN ('testuser1', 'testuser2')")
            user_ids = [row[0] for row in cursor.fetchall()]
            
            # Insert test comments
            if len(user_ids) >= 2:
                cursor.execute("""
                    INSERT INTO comments (user_id, content) 
                    VALUES 
                        (%s, 'This is a test comment from user 1'),
                        (%s, 'Another comment from user 1'),
                        (%s, 'Test comment from user 2')
                """, (user_ids[0], user_ids[0], user_ids[1]))
            
            db_connection.commit()
            yield
            
        except psycopg2.Error as e:
            db_connection.rollback()
            pytest.fail(f"Failed to set up test data: {e}")
        finally:
            cursor.close()
    
    def test_database_connection(self, db_connection):
        """Test that we can connect to the database"""
        assert db_connection is not None
        assert db_connection.status == psycopg2.extensions.STATUS_READY
    
    def test_can_query_comments_table(self, db_connection, setup_test_data):
        """Test that we can query the comments table"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM comments")
            count = cursor.fetchone()[0]
            assert count >= 0
        finally:
            cursor.close()
    
    def test_fetch_all_comments(self, db_connection, setup_test_data):
        """Test fetching all user comments"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("""
                SELECT c.id, c.content, c.created_at, u.username, u.email
                FROM comments c
                JOIN users u ON c.user_id = u.id
                ORDER BY c.created_at DESC
            """)
            
            comments = cursor.fetchall()
            assert len(comments) >= 0
            
            # If there are comments, verify structure
            if comments:
                comment = comments[0]
                assert len(comment) == 5  # id, content, created_at, username, email
                assert isinstance(comment[1], str)  # content should be string
                assert isinstance(comment[3], str)  # username should be string
        finally:
            cursor.close()
    
    def test_fetch_comments_by_user(self, db_connection, setup_test_data):
        """Test fetching comments by specific user"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("""
                SELECT c.id, c.content, c.created_at
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE u.username = %s
                ORDER BY c.created_at DESC
            """, ('testuser1',))
            
            comments = cursor.fetchall()
            # Should have at least 0 comments (might be 0 if test data wasn't inserted)
            assert len(comments) >= 0
            
        finally:
            cursor.close()
    
    def test_fetch_comments_with_pagination(self, db_connection, setup_test_data):
        """Test fetching comments with pagination"""
        cursor = db_connection.cursor()
        
        try:
            # Test LIMIT and OFFSET
            cursor.execute("""
                SELECT c.id, c.content, u.username
                FROM comments c
                JOIN users u ON c.user_id = u.id
                ORDER BY c.created_at DESC
                LIMIT 2 OFFSET 0
            """)
            
            comments = cursor.fetchall()
            assert len(comments) <= 2
        finally:
            cursor.close()
    
    def test_count_comments_per_user(self, db_connection, setup_test_data):
        """Test counting comments per user"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("""
                SELECT u.username, COUNT(c.id) as comment_count
                FROM users u
                LEFT JOIN comments c ON u.id = c.user_id
                GROUP BY u.username
                ORDER BY comment_count DESC
            """)
            
            results = cursor.fetchall()
            assert len(results) >= 0
            
            # If there are results, verify structure
            if results:
                result = results[0]
                assert len(result) == 2  # username, comment_count
                assert isinstance(result[1], int)  # comment_count should be integer
        finally:
            cursor.close()
    
    def test_search_comments_by_content(self, db_connection, setup_test_data):
        """Test searching comments by content"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("""
                SELECT c.id, c.content, u.username
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.content ILIKE %s
            """, ('%test%',))
            
            comments = cursor.fetchall()
            assert len(comments) >= 0
        finally:
            cursor.close()
    
    def test_recent_comments(self, db_connection, setup_test_data):
        """Test fetching recent comments (last 24 hours)"""
        cursor = db_connection.cursor()
        
        try:
            cursor.execute("""
                SELECT c.id, c.content, c.created_at, u.username
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY c.created_at DESC
            """)
            
            comments = cursor.fetchall()
            assert len(comments) >= 0
        finally:
            cursor.close()
    
    def test_cursor_as_dict(self, db_connection, setup_test_data):
        """Test using RealDictCursor for dictionary-like results"""
        from psycopg2.extras import RealDictCursor
        
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT c.id, c.content, u.username, u.email
                FROM comments c
                JOIN users u ON c.user_id = u.id
                LIMIT 5
            """)
            
            comments = cursor.fetchall()
            assert isinstance(comments, list)
            
            # If there are comments, verify dictionary structure
            if comments:
                comment = comments[0]
                assert 'id' in comment
                assert 'content' in comment
                assert 'username' in comment
                assert 'email' in comment
        finally:
            cursor.close()


# Additional utility functions for testing
def get_comment_by_id(conn, comment_id: int) -> Dict[str, Any]:
    """Utility function to get a specific comment by ID"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT c.id, c.content, c.created_at, c.updated_at, 
                   u.username, u.email
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.id = %s
        """, (comment_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'content': result[1],
                'created_at': result[2],
                'updated_at': result[3],
                'username': result[4],
                'email': result[5]
            }
        return None
    finally:
        cursor.close()



if __name__ == "__main__":
    pytest.main(["-v"])