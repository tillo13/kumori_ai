import logging
from datetime import datetime
from psycopg2.extras import DictCursor
import psycopg2.extras



logging.basicConfig(
    level=logging.DEBUG, # Log messages at DEBUG level and higher
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import psycopg2
from os import environ as env
from .google_secret_utils import get_secret_version

# Function to get database credentials from Google Cloud Secret Manager
def get_postgres_credentials(gcp_project_id):
    return {
        'host': get_secret_version(gcp_project_id, 'KUMORI_POSTGRES_IP'),
        'dbname': get_secret_version(gcp_project_id, 'KUMORI_POSTGRES_DB_NAME'),
        'user': get_secret_version(gcp_project_id, 'KUMORI_POSTGRES_USERNAME'),
        'password': get_secret_version(gcp_project_id, 'KUMORI_POSTGRES_PASSWORD'),
        'connection_name': get_secret_version(gcp_project_id, 'KUMORI_POSTGRES_CONNECTION_NAME'),
    }

# Function to create a database connection
def get_db_connection(gcp_project_id):
    db_credentials = get_postgres_credentials(gcp_project_id)
    # Check if the GAE_ENV environment variable is set, indicating the app is running on GCP
    is_gcp = env.get('GAE_ENV', '').startswith('standard')

    # GCP environment
    if is_gcp:
        db_socket_dir = env.get("DB_SOCKET_DIR", "/cloudsql")
        cloud_sql_connection_name = db_credentials['connection_name']
        host = f"{db_socket_dir}/{cloud_sql_connection_name}"
    else:
        # Local environment
        host = db_credentials['host']
    
    # Create and return the connection
    conn = psycopg2.connect(
        dbname=db_credentials['dbname'],
        user=db_credentials['user'],
        password=db_credentials['password'],
        host=host
    )
    return conn


###############################################2024mar9 get user data for profile###############################################
def get_user_social_posts(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    social_posts = []
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT * 
        FROM prod_user_social_posts p
        INNER JOIN prod_user_accounts ua ON p.user_account_id = ua.pk_id
        WHERE ua.auth_provider_id = %s
        ORDER BY p.created_at DESC;  
        """
        
        cursor.execute(query, (auth_provider_id,))
        posts_records = cursor.fetchall()
        
        social_posts = [dict(record) for record in posts_records]  # Convert records into a list of dictionaries

    except Exception as e:
        logging.error("Database error while fetching user social posts: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return social_posts

def get_user_topics(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    topics = []
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Include nt.pk_id in your SELECT query
        query = """
        SELECT nt.pk_id, nt.topic, nt.created_at 
        FROM prod_user_news_topics nt
        INNER JOIN prod_user_accounts ua ON nt.user_account_id = ua.pk_id
        WHERE ua.auth_provider_id = %s AND nt.isActive = TRUE
        ORDER BY nt.pk_id DESC;  
        """
        
        cursor.execute(query, (auth_provider_id,))
        topics_records = cursor.fetchall()
        
        # Include pk_id in the returned topics dictionaries
        topics = [
            {
                'pk_id': record['pk_id'],  # Include this row
                'topic': record['topic'], 
                'created_at': record['created_at']
            } 
            for record in topics_records
        ]

    except Exception as e:
        logging.error("Database error while fetching user news topics: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return topics

def deactivate_topic(gcp_project_id, user_auth_provider_id, pk_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        # Update isActive to FALSE for the specific topic
        cursor.execute("""
            UPDATE prod_user_news_topics
            SET isActive = FALSE
            WHERE pk_id = %s
            AND user_account_id = (
                SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s
            );
        """, (pk_id, user_auth_provider_id))
        
        conn.commit()
        
        # Log information about the deactivation
        logging.info(f"Topic with pk_id={pk_id} deactivated for user {user_auth_provider_id}.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while deactivating topic: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_user_writing_styles(gcp_project_id, user_auth_provider_id):
    conn = None
    cursor = None
    writing_styles = []
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Updated query to filter only active writing styles
        query = """
        SELECT DISTINCT ON (style_content) prod_user_writing_styles.pk_id, style_content, prod_user_writing_styles.created_at
        FROM prod_user_writing_styles
        INNER JOIN prod_user_accounts ON prod_user_writing_styles.user_account_id = prod_user_accounts.pk_id
        WHERE prod_user_accounts.auth_provider_id = %s AND prod_user_writing_styles.isActive = TRUE
        ORDER BY style_content, prod_user_writing_styles.pk_id DESC;
        """
        
        cursor.execute(query, (user_auth_provider_id,))
        records = cursor.fetchall()
        
        writing_styles = [
            {'style_id': record['pk_id'], 'style_content': record['style_content'], 'created_at': record['created_at']} 
            for record in records
        ]

    except Exception as e:
        logging.error("Database error while fetching user writing styles: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return writing_styles
    
    return writing_styles
def deactivate_user_writing_style(gcp_project_id, user_auth_provider_id, pk_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        # Assuming there's an isActive column in your prod_user_writing_styles table
        cursor.execute("""
            UPDATE prod_user_writing_styles
            SET isActive = FALSE
            WHERE pk_id = %s
            AND user_account_id = (
                SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s
            );
        """, (pk_id, user_auth_provider_id))
        
        conn.commit()
        
        # Log the deactivation
        logging.info(f"Writing style with pk_id={pk_id} deactivated for user {user_auth_provider_id}.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while deactivating writing style: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
###############################################2024mar9 get user data for profile###############################################

###############################################2024mar9 new user logs###############################################
def save_user_style(gcp_project_id, user_auth_provider_id, user_style):
    conn = None
    cursor = None
    pk_id = None  # Add this line to initialize the pk_id variable
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        # Retrieve user_account_id using the auth_provider_id
        cursor.execute("SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s", (user_auth_provider_id,))
        user_account_record = cursor.fetchone()
        
        if user_account_record:
            user_account_id = user_account_record[0]
            
            # Insert the user's style into the appropriate table (assuming a table schema here)
            cursor.execute("""
                INSERT INTO prod_user_writing_styles (user_account_id, style_content, created_at, updated_at)
                VALUES (%s, %s, %s, %s) RETURNING pk_id;
            """, (user_account_id, user_style, datetime.now(), datetime.now()))
            pk_id = cursor.fetchone()[0]  # Retrieve the returned pk_id from the INSERT statement
            conn.commit()
            logging.info('User style saved for user account ID: %s', user_account_id)
            
        else:
            logging.error('No user account found for auth_provider_id: %s', user_auth_provider_id)

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while saving user style: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return pk_id  # Return the pk_id at the end of the function

def save_ai_generated_linkedin_post(gcp_project_id, user_auth_provider_id, openai_response, model_identifier, style_id, platform=None, article_title=None):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        # Get the user_account_id from the auth_provider_id
        cursor.execute("SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s", (user_auth_provider_id,))
        user_account_record = cursor.fetchone()
        
        if not user_account_record:
            logging.error('No user account found for auth_provider_id: %s', user_auth_provider_id)
            return
        
        user_account_id = user_account_record[0]

        # Updated the SQL query to include style_id
        cursor.execute("""
            INSERT INTO prod_user_social_posts (
                user_account_id, style_id, generated_text, model_identifier, platform, article_title, post_timestamp, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (user_account_id, style_id, openai_response, model_identifier, platform, article_title, datetime.now(), datetime.now(), datetime.now()))

        conn.commit()
        logging.info('Generated text saved with style_id for user account ID: %s', user_account_id)
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while saving generated text with style_id: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_user_topics(gcp_project_id, user_auth_provider_id, topics):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # First, get the pk_id for the corresponding auth_provider_id from prod_user_accounts
        cursor.execute("SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s", (user_auth_provider_id,))
        user_account = cursor.fetchone()
        
        if not user_account:
            logging.error('No user account found for auth_provider_id: %s', user_auth_provider_id)
            return

        user_account_id = user_account[0]

        # Go through each topic and check if it should be inserted
        for topic in topics:
            # Skip if the topic is blank
            if not topic:
                logging.info('Skipping blank topic for user account ID: %s', user_account_id)
                continue
            
            # Check if the topic is already saved for the user
            cursor.execute("""
                SELECT COUNT(*) FROM prod_user_news_topics 
                WHERE user_account_id = %s AND topic = %s;
            """, (user_account_id, topic))
            if cursor.fetchone()[0] > 0:
                logging.info('Topic "%s" is already saved for user account ID: %s', topic, user_account_id)
                continue

            # Insert the new topic
            cursor.execute("""
                INSERT INTO prod_user_news_topics (user_account_id, topic, updated_at) 
                VALUES (%s, %s, %s);
            """, (user_account_id, topic, datetime.now()))
            logging.info('Topic "%s" saved for user account ID: %s', topic, user_account_id)

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while saving user topics: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def save_user_news_article(gcp_project_id, article_data, auth_provider_id):  # auth_provider_id is the string Auth0 ID
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        # First, get the pk_id for the corresponding auth_provider_id from prod_user_accounts
        cursor.execute("SELECT pk_id FROM prod_user_accounts WHERE auth_provider_id = %s", (auth_provider_id,))
        user_account = cursor.fetchone()
        
        if not user_account:
            logging.error('No user account found for auth_provider_id: %s', auth_provider_id)
            return
        
        user_account_id = user_account[0]  # This is the bigint you need for user_account_id

        # Now, proceed with the original insert statement using user_account_id bigint
        insert_stmt = """
            INSERT INTO prod_user_news_articles (
                title, summary, published_date, source, url, keyword, category, user_account_id, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING pk_id;
        """

        # Data for the SQL insert statement, including the correct bigint user_account_id
        insert_data = (
            article_data['title'],
            article_data['description'],  # Assuming 'description' is where the article's summary is stored
            article_data['published_at'],
            article_data['source'],
            article_data['url'],
            article_data['keyword'],
            article_data['category'],
            user_account_id,  # Bigint user_account_id from prod_user_accounts
            datetime.now(),  # Use `datetime.utcnow()` for UTC time
        )
        
        # Execute the SQL insert statement
        cursor.execute(insert_stmt, insert_data)
        article_id = cursor.fetchone()[0]  # Get the primary key of the inserted article
        conn.commit()
        logging.info('Article with ID %s saved successfully for user account ID: %s', article_id, user_account_id)
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while saving user news article: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to insert a visitor log entry
def log_visitor(gcp_project_id, user_metadata):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)  
        cursor = conn.cursor()

        # SQL insert statement
        insert_stmt = """
            INSERT INTO visitor_log (
                family_name, given_name, full_name, nickname, picture_url, auth_provider_id, last_update
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        # Data for the SQL insert statement
        insert_data = (
            user_metadata.get('family_name'),
            user_metadata.get('given_name'),
            user_metadata.get('name'),
            user_metadata.get('nickname'),
            user_metadata.get('picture'),
            user_metadata.get('sub'),
            user_metadata.get('updated_at'),
        )

        # Execute the SQL insert statement
        cursor.execute(insert_stmt, insert_data)
        conn.commit()
        logging.info('Visitor log entry successfully created for user %s (%s, %s).', user_metadata.get('sub'), user_metadata.get('name'), user_metadata.get('nickname'))

    except Exception as e:
        if conn is not None:
            conn.rollback()
        logging.error("Database error while logging visitor: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to get the visit count and the previous last login timestamp for a specific auth_provider_id
def get_user_visit_data(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # SQL query to fetch the number of visits and the timestamp of the second most recent visit
        query_stmt = """
            SELECT COUNT(*), (
                SELECT last_update
                FROM visitor_log
                WHERE auth_provider_id = %s
                ORDER BY last_update DESC
                OFFSET 1
                LIMIT 1
            ) AS previous_last_login
            FROM visitor_log
            WHERE auth_provider_id = %s
        """

        # Execute the SQL query with the auth_provider_id used for both subquery and main query
        cursor.execute(query_stmt, (auth_provider_id, auth_provider_id))
        result = cursor.fetchone()
        # If there are no results, default to 0 visits and None for the previous last login timestamp
        visit_count, previous_last_login_timestamp = result if result else (0, None)

        # If the visit count is 1 or 0, then there is no previous last login timestamp
        if visit_count <= 1:
            previous_last_login_timestamp = None

        return visit_count, previous_last_login_timestamp

    except Exception as e:
        logging.error("Database error while fetching user visit data: %s", e)
        return 0, None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to upsert user account data and retrieve user info
def upsert_user_account_and_get_info(gcp_project_id, user_metadata):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        # Use DictCursor to get dictionary-like cursor which allows column access by name
        cursor = conn.cursor(cursor_factory=DictCursor)


        # SQL statement to INSERT a new user OR UPDATE if the user already exists based on the auth_provider_id
        # The RETURNING clause will return the data after the insert or update is executed
        upsert_stmt = """
            INSERT INTO prod_user_accounts (
                auth_provider_id, last_name, first_name, nickname, full_name, email_address, profile_picture, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (auth_provider_id) DO UPDATE SET
                last_name = EXCLUDED.last_name,
                first_name = EXCLUDED.first_name,
                nickname = EXCLUDED.nickname,
                full_name = EXCLUDED.full_name,
                email_address = EXCLUDED.email_address,
                profile_picture = EXCLUDED.profile_picture,
                updated_at = EXCLUDED.updated_at
            RETURNING *;
        """

        # Define the data to insert or update in the table
        upsert_data = (
            user_metadata['sub'],  # Assuming this is the unique auth_provider_id
            user_metadata.get('family_name'),
            user_metadata.get('given_name'),
            user_metadata.get('nickname'),
            user_metadata.get('name'),
            user_metadata.get('email'),
            user_metadata.get('picture'),
            datetime.now(),  # Current time as updated_at
        )

        # Execute the upsert statement
        cursor.execute(upsert_stmt, upsert_data)
        user_info = cursor.fetchone()  # Fetch the result of the UPSERT operation as a dictionary-like object
        conn.commit()
        
        # Since user_info is now a dictionary-like object, we can access values using keys
        logging.info('User account upserted successfully for user: %s', user_info['auth_provider_id'])

        # Return the user info directly; no need to convert to dict since it's already dictionary-like
        return user_info

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Database error while upserting user account: %s", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

###############################################2024mar9 new user logs###############################################











# Function to fetch a single image entry by `pk_id` and `user_id`
def select_single_image(gcp_project_id, pk_id, user_id):
    # Use the established get_db_connection function
    conn = get_db_connection(gcp_project_id)
    cursor = conn.cursor(cursor_factory=DictCursor)  # Use DictCursor to get a dictionary
    try:
        cursor.execute("""
            SELECT *
            FROM log_dalle_images
            WHERE pk_id = %s AND user_id = %s;
        """, (pk_id, user_id,))
        image_data = cursor.fetchone()
        return dict(image_data) if image_data else None
    except Exception as e:
        logging.error("Database error while fetching image details: %s", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
def fetch_user_videos_with_screenshots_filtered(gcp_project_id, auth_provider_id=None, video_id=None):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor(cursor_factory=DictCursor)

        # Start by defining a condition for non-NULL overall_gcs_image_url
        conditions = ["overall_gcs_image_url IS NOT NULL"]

        # Define the SELECT statement with the correct column aliases
        query = """
        SELECT DISTINCT ON (uvc_submitter_auth_provider_id, uvc_video_id)
            uvc_submitter_auth_provider_id AS auth_provider_id,
            uvc_video_id AS video_id,
            uvc_video_url AS video_url,
            uvc_video_title AS video_title,
            uvc_submitter_submission_timestamp AS submission_timestamp,
            uvs_completion_timestamp AS completion_timestamp,
            uvc_gpt_vision_description AS gpt_vision_description,
            uvc_dalle3_revised_prompt AS dalle3_revised_prompt,
            overall_gcs_image_url,
            ldi_size AS ldi_size,
            ldi_style AS ldi_style,
            ldi_quality AS ldi_quality,
            uvc_screenshot_time AS screenshot_time,
            uvc_video_caption AS video_caption,
            youtube_link_at_screenshot_time AS youtube_link_at_screenshot_time
        FROM
            vw_youtube_screencaps_successful_creations
        """

        # Initialize a list to store parameter values for the WHERE clause conditions.
        params = []

        # Append a condition for filtering by auth_provider_id and video_id if provided.
        if auth_provider_id:
            conditions.append("uvc_submitter_auth_provider_id = %s")
            params.append(auth_provider_id)

        if video_id:
            conditions.append("uvc_video_id = %s")
            params.append(video_id)

        # Add the WHERE clause to the query if conditions were specified.
        query += " WHERE " + " AND ".join(conditions)

        # Add the ORDER BY clause to ensure consistent query results.
        query += " ORDER BY uvc_submitter_auth_provider_id, uvc_video_id, submission_timestamp DESC"

        # Log the final query and parameters to debug any potential issues.
        logging.info("Executing query: %s with params: %s", query, params)

        # Execute the query using the parameters.
        cursor.execute(query, params)
        
        # Fetch all the results and return them as a list of dictionaries.
        videos_data = cursor.fetchall()
        return videos_data

    except Exception as e:
        # If an exception occurs, log it as an error with the exception message.
        logging.error("Database error while fetching user video data: %s", e)
        # If there's an error, return an empty list to indicate no data was retrieved.
        return []
    finally:
        # Safely close the cursor and connection if they were opened.
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def fetch_simplified_videos_with_screenshots(gcp_project_id, video_id=None):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT DISTINCT ON (overall_gcs_image_url) 
            overall_gcs_image_url, uvc_author, uvc_submitter_auth_provider_id, uvs_auth_provider_id, 
            uvs_model_choice, youtube_link_at_screenshot_time, uvc_video_url, uvc_video_title,
            uvc_video_id, uvc_video_length, uvc_views, uvc_screenshot_filename, uvc_screenshot_time,
            uvc_video_caption, uvc_gpt_vision_description, uvc_dalle3_revised_prompt, ldi_size,
            ldi_style, ldi_quality, uvc_submitter_submission_timestamp, uvs_completion_timestamp, uvs_model_choice, uvs_pk_id, uvs_completion_epoch
        FROM 
            vw_youtube_screencaps_successful_creations
        WHERE 
            overall_gcs_image_url IS NOT NULL
        """
        params = []

        # Apply the filter only if the video_id is provided
        if video_id:
            query += " AND uvc_video_id = %s"
            params.append(video_id)

        # Add an ORDER BY clause because DISTINCT ON requires the rows to be ordered by the distinct column(s)
        query += " ORDER BY overall_gcs_image_url;"

        logging.info("Executing query: %s with params: %s", query, params)
        
        cursor.execute(query, params)
        videos_data = cursor.fetchall()
        return videos_data
        
    except Exception as e:
        logging.error("Database error while fetching simplified videos with screenshots: %s", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



#2023jan22 I think this one can go away...
def fetch_user_videos_with_screenshots(gcp_project_id, auth_provider_id=None, video_id=None):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # Start the query with a condition that's always true
        query = """
        SELECT * FROM vw_youtube_screencap_ai_creations
        WHERE true
        """
        params = []

        # Apply additional filters only if the parameters are given
        if auth_provider_id:
            query += " AND uvc_submitter_auth_provider_id = %s"
            params.append(auth_provider_id)
        if video_id:
            query += " AND uvc_video_id = %s"
            params.append(video_id)

        query += ";"

        # Executing the query with the proper parameters
        cursor.execute(query, params)
        videos_data = cursor.fetchall()

        
        videos_and_screenshots = {}
        for row in videos_data:
            video_id = row[15]  # Index of uvc_video_id according to schema
            screenshot_details = {
                'ldi_pk_id': row[0],
                'ldi_user_id': row[1],
                'ldi_model': row[3],
                'ldi_n': row[4],
                'ldi_size': row[5],
                'ldi_response_format': row[6],
                'ldi_style': row[7],
                'ldi_quality': row[8],
                'ldi_created_at': row[11].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row[11], datetime) else None,
                'ldi_gcp_image_url': row[12],
                'ldi_created_via': row[13],
                'uvc_pk_id': row[14],
                'uvc_video_id': row[15],
                'uvc_video_url': row[16],
                'uvc_video_title': row[17],
                'uvc_video_length': row[18],
                'uvc_views': row[19],
                'uvc_author': row[20],
                'uvc_screenshot_filename': row[21],
                'uvc_screenshot_time': row[22],
                'uvc_video_caption': row[23],
                'uvc_caption_start': row[24],
                'uvc_caption_duration': row[25],
                'uvc_screenshot_captured_at': row[26],
                'uvc_gpt_vision_updated_at': row[27],
                'uvc_gpt_vision_description': row[28],
                'uvc_dalle3_updated_at': row[29],
                'uvc_dalle3_response_url': row[30],
                'uvc_dalle3_revised_prompt': row[31],
                'uvc_dalle3_created_epoch': row[32],
                'uvc_processed_at': row[33],
                'uvc_submitter_pk_id': row[34],
                'uvc_submitter_auth_provider_id': row[35],
                'uvc_submitter_submission_timestamp': row[36],
                'uvs_pk_id': row[37],
                'uvs_auth_provider_id': row[38],
                'uvs_video_url': row[39],
                'uvs_submission_timestamp': row[40],
                'uvs_binary_completion': row[41],
                'uvs_completion_timestamp': row[42],
                'uvs_updated_by': row[43],
                'youtube_link_at_screenshot_time': row[44],
                'uvs_model_choice': row[45],
                'uvs_completion_epoch': row[46]
            }

            
            if video_id not in videos_and_screenshots:
                videos_and_screenshots[video_id] = []


            # uvc_submitter_submission_timestamp
            submitter_submission_timestamp = row[36]  # Assuming this is the correct index
            if submitter_submission_timestamp and submitter_submission_timestamp != 'No Submission Timestamp':
                # Convert the submitter submission timestamp to a string formatted datetime
                submitter_submission_timestamp = submitter_submission_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # If there is no submission timestamp, keep the placeholder string
                submitter_submission_timestamp = 'No Submission Timestamp'

            screenshot_details['uvc_submitter_submission_timestamp'] = submitter_submission_timestamp

            
            videos_and_screenshots[video_id].append(screenshot_details)

        return videos_and_screenshots

    except Exception as e:
        logging.error("Database error while fetching user videos with screenshots: %s", e)
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
def save_user_video_submission(gcp_project_id, auth_provider_id, video_url, model_choice, binary_completion=0):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        insert_stmt = """
            INSERT INTO user_video_submissions (
                auth_provider_id, video_url, model_choice, binary_completion
            ) VALUES (%s, %s, %s, %s) RETURNING pk_id;
        """
        cursor.execute(insert_stmt, (auth_provider_id, video_url, model_choice, binary_completion))
        pk_id = cursor.fetchone()[0]
        conn.commit()
        logging.info("New user video submission saved successfully with pk_id %s", pk_id)
        return pk_id

    except Exception as e:
        logging.error("Error saving new user video submission: %s", e)
        if conn:
            conn.rollback()
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_leaderboard_data(gcp_project_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # Ensure that the SELECT query is matching the order of the columns in your view or specify them explicitly
        cursor.execute('SELECT auth_provider_id, user_email, points, pseudonym, last_interaction, ranking FROM vw_leaderboard;')

        leaderboard_entries = []
        for row in cursor.fetchall():
            leaderboard_entries.append({
                "auth_provider_id": row[0],
                "user_email": row[1],
                "points": row[2],
                "pseudonym": row[3],
                "last_interaction": row[4],
                "ranking": row[5] 
            })

        return leaderboard_entries
    except Exception as e:
        logging.error("Database error while fetching leaderboard data: %s", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# This function gets the vistor and user counts based on the auth_provider_id
def get_vw_load_dashboard_metadata(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)  # Assumes you already have this function defined to get a db connection
        cursor = conn.cursor()
        
        # Define the SQL statement to select from the view for a specific auth_provider_id
        query_stmt = """
            SELECT
                visit_count,
                last_login_timestamp,
                user_content_count,
                dalle_image_count,
                total_days_since_first_seen,
                total_engagements,
                avg_daily_engagements
            FROM
                vw_load_dashboard_metadata
            WHERE
                auth_provider_id = %s;
        """
        
        # Execute the SQL query with the auth_provider_id
        cursor.execute(query_stmt, (auth_provider_id,))

        # Fetch the results
        result = cursor.fetchone()

        if not result:
            return {}

        # Map the results to a dictionary
        dashboard_metadata = {
            "visit_count": result[0],
            "last_login": result[1],  # You may need to format this timestamp
            "user_content_count": result[2],
            "dalle_image_count": result[3],
            "total_days_since_first_seen": result[4],
            "total_engagements": result[5],
            "avg_daily_engagements": result[6],
        }

        return dashboard_metadata

    except Exception as e:
        logging.error("Database error while fetching dashboard metadata: %s", e)
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# This function retrieves the dashboard data based on the auth_provider_id --has some data 2023jan05 but not sure we'll use it yet, keep for now.
def get_load_dashboard(gcp_project_id, auth_provider_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM vw_load_dashboard WHERE auth_provider_id = %s", (auth_provider_id,))
            result = cursor.fetchone()
            if result:
                # Convert to dictionary if dashboard data exists
                column_names = [desc[0] for desc in cursor.description]
                return dict(zip(column_names, result))
            else:
                return None
    finally:
        conn.close()

# This function retrieves the user profile and related stats based on the auth_provider_id
def get_load_profile(gcp_project_id, auth_provider_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM vw_load_profile WHERE auth_provider_id = %s", (auth_provider_id,))
            result = cursor.fetchone()
            if result:
                column_names = [desc[0] for desc in cursor.description]
                user_profile = dict(zip(column_names, result))
                return user_profile
            else:
                return None
    finally:
        conn.close()





# Function to fetch DALL-E image generation history for a specific user
def get_dalle_image_history(gcp_project_id, user_id, include_cost=False):
    conn = None
    cursor = None
    try:
        # Establish a database connection
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # SQL query to retrieve all DALL-E image log entries for the user
        query_stmt = """
            SELECT *
            FROM log_dalle_images
            WHERE user_id = %s
            ORDER BY pk_id DESC;
        """
        
        # Execute the query with the user_id
        cursor.execute(query_stmt, (user_id,))
        
        # Fetch all the rows matching the user_id
        results = cursor.fetchall()
        
        # Get the column names to map to the dictionary keys
        column_names = [desc[0] for desc in cursor.description]
        history_entries = [dict(zip(column_names, row)) for row in results]
        
        # calc current total cost
        total_cost = 0
        for entry in history_entries:
            if include_cost:
                cost = calculate_image_cost(entry)
                entry['cost'] = cost
                total_cost += cost

        return (history_entries, total_cost) if include_cost else history_entries

    except Exception as e:
        logging.error("Database error while fetching DALL-E image history: %s", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Add a function to calculate the cost of images based on pricing
def calculate_image_cost(entry):
    pricing_dict = {
        '1024x1024': {'standard': 0.040, 'hd': 0.080},
        '1792x1024': {'standard': 0.080, 'hd': 0.120},
        '1024x1792': {'standard': 0.080, 'hd': 0.120},
    }
    
    # Using .get() to retrieve the cost and defaulting to 0 if not found
    size_price = pricing_dict.get(entry['size'], {}).get(entry['quality'], 0)
    return size_price








# This function retrieves the user profile based on the auth_provider_id
def get_user_profile(gcp_project_id, auth_provider_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM user_profile WHERE auth_provider_id = %s", (auth_provider_id,))
            result = cursor.fetchone()
            if result:
                # Convert to dictionary if profile exists
                column_names = [desc[0] for desc in cursor.description]
                return dict(zip(column_names, result))
            else:
                return None
    finally:
        conn.close()



def get_gptvision_costs_per_user(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        query_stmt = """
            SELECT *
            FROM vw_openai_data_per_user_gptvision
            WHERE auth_provider_id = %s;
        """
        cursor.execute(query_stmt, (auth_provider_id,))
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        gptvision_cost_data = [dict(zip(column_names, row)) for row in result]

        return gptvision_cost_data

    except Exception as e:
        logging.error("Database error while fetching GPTVISION costs per user: %s", e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def get_dalle_costs_per_user(gcp_project_id, auth_provider_id): 
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        query_stmt = """
            SELECT *
            FROM vw_openai_data_per_user_dalle
            WHERE auth_provider_id = %s;
        """
        cursor.execute(query_stmt, (auth_provider_id,))
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        dalle_cost_data = [dict(zip(column_names, row)) for row in result]

        return dalle_cost_data

    except Exception as e:
        logging.error("Database error while fetching DALL-E costs per user: %s", e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_chatgpt_costs_per_user(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        query_stmt = """
            SELECT *
            FROM vw_openai_data_per_user_chatgpt
            WHERE auth_provider_id = %s;
        """
        cursor.execute(query_stmt, (auth_provider_id,))
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        chatgpt_cost_data = [dict(zip(column_names, row)) for row in result]

        return chatgpt_cost_data

    except Exception as e:
        logging.error("Database error while fetching ChatGPT costs per user: %s", e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()






def save_try_it_now_content(gcp_project_id, user_email, auth_provider_id, article_data,
                            writing_sample='awaiting_ai_submission', 
                            writing_style_analysis='awaiting_ai_submission', 
                            content_post='awaiting_ai_submission',
                            openai_model='undetermined'): 
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        insert_stmt = """
            INSERT INTO user_content (
                user_email, auth_provider_id, writing_sample, 
                writing_style_analysis, content_post,
                article_description, article_title, article_url,
                industry, content_objective, content_platform, 
                brand_tone, openai_model  
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING pk_id; 
        """

        insert_values = (
            user_email,
            auth_provider_id,
            writing_sample,
            writing_style_analysis,
            content_post,
            article_data.get('description', 'awaiting_ai_submission'),
            article_data.get('title', 'awaiting_ai_submission'),
            article_data.get('url', 'awaiting_ai_submission'),
            article_data.get('industry', 'awaiting_ai_submission'),
            article_data.get('content_objective', 'awaiting_ai_submission'),
            article_data.get('content_platform', 'linkedin'),  
            article_data.get('brand_tone', 'awaiting_ai_submission'),
            openai_model 
        )

        cursor.execute(insert_stmt, insert_values)
        pk_id = cursor.fetchone()[0]  

        conn.commit()
        logging.info("New try_it_now content saved successfully with pk_id %s", pk_id)
        return pk_id

    except Exception as e:
        logging.error("Error saving try_it_now content to user_content table: %s", e)
        if conn:
            conn.rollback()
        raise  

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()




def update_try_it_now_user_content(gcp_project_id, pk_id, content_post, user_email, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        
        update_stmt = """
            UPDATE user_content
            SET content_post = %s, user_email = %s, auth_provider_id = %s
            WHERE pk_id = %s;
        """
        
        cursor.execute(update_stmt, (content_post, user_email, auth_provider_id, pk_id))
        conn.commit()

        affected_rows = cursor.rowcount
        logging.info("User content updated successfully for pk_id %s with %s affected rows.", pk_id, affected_rows)

    except Exception as e:
        if conn is not None:
            conn.rollback()
        logging.exception("Error updating user content for PK ID %s: %s", pk_id, e)
        raise  # Re-raise the exception to handle it upstream if necessary

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()







def fetch_model_cost(gcp_project_id, model_name_alias, cost_type):
    model_map = {
        'gpt-4-vision-preview': 'gpt-4-1106-vision-preview'  # Map alias to actual model name in the DB
    }
    actual_model_name = model_map.get(model_name_alias, model_name_alias)
    
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT cost_per_1k_tokens FROM openai_pricing 
                WHERE model = %s AND cost_type = %s AND is_latest = true
            """, (actual_model_name, cost_type))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return None
    finally:
        if conn is not None:
            conn.close()



def log_openai_usage(gcp_project_id, auth_provider_id, product_type, model, tokens_used, cost, prompt_excerpt='', resource_identifier=''):
    conn = get_db_connection(gcp_project_id)
    pk_id = None  # Initialize pk_id to None

    # Default values for unknown product_type and model
    DEFAULT_PRODUCT_TYPE = "unknown"
    DEFAULT_MODEL = "unknown"

    # Use default values if cost indicates an unknown model and provided values are not specified
    if cost == 0.00 and not product_type:
        product_type = DEFAULT_PRODUCT_TYPE
    if cost == 0.00 and not model:
        model = DEFAULT_MODEL

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO openai_usage_log (
                    auth_provider_id, product_type, model, tokens_used, cost, prompt_excerpt, resource_identifier
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING pk_id;
            """, (auth_provider_id, product_type, model, tokens_used, cost, prompt_excerpt[:255], resource_identifier[:255]))
            # Fetch the pk_id only if a row was inserted
            result = cursor.fetchone()
            pk_id = result[0] if result else None
            conn.commit()
            
            # Log a message with the pk_id or if it was a default cost
            if pk_id is not None:
                logging.info(f"OpenAI usage logged with pk_id: {pk_id}")
            else:
                logging.warning(f"OpenAI usage logged with default cost for model: {model}, no pk_id returned.")
            return pk_id
    except Exception as e:
        logging.error(f"Failed to log OpenAI usage: {e}")
        raise e
    finally:
        if conn is not None:
            conn.close()





# Function to log DALL-E image generation details and return the primary key id of the logged record
def log_dalle_image(gcp_project_id, user_id, prompt, model, n, size, response_format, style, quality):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        # SQL command to insert a log entry with returning the primary key id
        insert_stmt = """
            INSERT INTO log_dalle_images (
                user_id, prompt, model, n, size, response_format, style, quality
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING pk_id;
        """
        insert_values = (user_id, prompt, model, n, size, response_format, style, quality)
        cursor.execute(insert_stmt, insert_values)
        # Get the primary key id of the new record
        pk_id = cursor.fetchone()[0]
        conn.commit()
        logging.info("Successfully logged DALL-E image creation attempt.")
        return pk_id  # Return the primary key id
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("Failed to log DALL-E image creation attempt.", exc_info=error)
        if conn is not None:
            conn.rollback()
        return None  # If error, return None
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()










# Change the function definition to include the gcp_image_url parameter
def update_log_dalle_via_pk_id(gcp_project_id, pk_id, image_url=None, revised_prompt=None, gcp_image_url=None):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # Prepare the SQL statement to update the provided column values
        # Make sure to remove Python-style comments (#) before running SQL statements
        update_stmt = """
            UPDATE log_dalle_images
            SET image_url = COALESCE(%s, image_url),
                revised_prompt = COALESCE(%s, revised_prompt),
                gcp_image_url = COALESCE(%s, gcp_image_url)
            WHERE pk_id = %s;
        """
        update_values = (image_url, revised_prompt, gcp_image_url, pk_id)
        
        cursor.execute(update_stmt, update_values)
        conn.commit()
        logging.info("Successfully updated DALL-E image log with pk_id: %s", pk_id)
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("Failed to update DALL-E image log.", exc_info=error)
        if conn is not None:
            conn.rollback()
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()











def get_user_news_preferences(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        logging.debug(f"Fetching news preferences for user {auth_provider_id}")

        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        # Select the latest entry per user to get the most recent preferences
        query_stmt = """
            SELECT news_outlet_preferences, news_category_preferences, keywords
            FROM get_started
            WHERE auth_provider_id = %s
            ORDER BY last_updated DESC
            LIMIT 1
        """
        cursor.execute(query_stmt, (auth_provider_id,))
        result = cursor.fetchone()
        return result if result else (None, None)
    except Exception as e:
        logging.error(f"Database error while fetching news preferences: {e}")
        return (None, None)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()





def update_user_profile(gcp_project_id, auth_provider_id, industry, objective, news_categories, excluded_news_sources, interested_news_keywords):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            update_stmt = """
                UPDATE user_profile
                SET
                    industry = %s,
                    objective = %s,
                    news_categories = %s,
                    excluded_news_sources = %s,
                    interested_news_keywords = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE auth_provider_id = %s;
            """
            cursor.execute(update_stmt, (industry, objective, news_categories, excluded_news_sources, interested_news_keywords, auth_provider_id,))
            conn.commit()
    finally:
        conn.close()

def update_user_stripe_customer_id(gcp_project_id, auth_provider_id, stripe_customer_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            update_stmt = """
                UPDATE user_profile
                SET stripe_customer_id = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE auth_provider_id = %s;
            """
            cursor.execute(update_stmt, (stripe_customer_id, auth_provider_id,))
            conn.commit()
    finally:
        conn.close()

def get_stripe_customer_id(gcp_project_id, auth_provider_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            query_stmt = """
                SELECT stripe_customer_id
                FROM user_profile
                WHERE auth_provider_id = %s;
            """
            cursor.execute(query_stmt, (auth_provider_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    finally:
        conn.close()





def get_user_content(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        query_stmt = """
            SELECT pk_id, user_email, auth_provider_id, writing_sample, writing_style_analysis, content_post,
                   article_description, article_title, article_url, created_at,
                   industry, content_objective, content_platform, brand_tone
            FROM user_content 
            WHERE auth_provider_id = %s
            ORDER BY pk_id DESC;
        """
        
        cursor.execute(query_stmt, (auth_provider_id,))
        results = cursor.fetchall()

        # Get column names to map to dictionary keys
        column_names = [desc[0] for desc in cursor.description]
        user_content_entries = [dict(zip(column_names, row)) for row in results]

        return user_content_entries
    except Exception as e:
        logging.error("Database error while fetching user content: %s", e)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def count_user_generated_dalle_images(gcp_project_id, user_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # SQL query to count the number of DALL·E image entries
        query_stmt = """
            SELECT COUNT(*)
            FROM log_dalle_images
            WHERE user_id = %s
        """

        # Execute the SQL query
        cursor.execute(query_stmt, (user_id,))
        result = cursor.fetchone()
        entry_count = result[0] if result else 0

        return entry_count

    except Exception as e:
        logging.error("Database error while counting DALL·E images: %s", e)
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



# This function creates a new user profile with just the auth_provider_id
def create_user_profile(gcp_project_id, auth_provider_id):
    conn = get_db_connection(gcp_project_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO user_profile (auth_provider_id) VALUES (%s)", (auth_provider_id,))
            conn.commit()
    finally:
        conn.close()



















# Function to insert a get_started entry --2023dec27 think I can drop these
def save_get_started_response(gcp_project_id, auth_provider_id, industry, objective, platform, keywords, tone, writing_sample, writing_style, news_outlet_preferences, news_category_preferences):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)  
        cursor = conn.cursor()

        # SQL insert statement for get_started, now including 'writing_style'
        insert_stmt = """
            INSERT INTO get_started (
                industry, content_objective, content_platform, keywords, brand_tone, vl_pk_id, auth_provider_id, writing_sample, writing_style, news_outlet_preferences, news_category_preferences
            ) VALUES (
                %s, %s, %s, %s, %s,
                (SELECT MAX(pk_id) FROM visitor_log WHERE auth_provider_id = %s),
                %s, %s, %s, %s, %s
            )
        """
        
        insert_data = (
            industry,
            objective,
            platform,
            keywords,
            tone,
            auth_provider_id,
            auth_provider_id,
            writing_sample,
            writing_style,
            news_outlet_preferences,
            news_category_preferences
        )

        # Execute the SQL insert statement
        cursor.execute(insert_stmt, insert_data)
        conn.commit()
        logging.info('*POSTGRES_UTILS.PY: Get started entry saved for auth_provider_id %s.', auth_provider_id)

        #return the value of the pk_id so that we can pass it to google cloud task to populate the writing_style asynchornously
        pk_id_cursor = conn.cursor()
        pk_id_cursor.execute("SELECT LASTVAL();")
        pk_id = pk_id_cursor.fetchone()[0]
        pk_id_cursor.close()
        return pk_id

    except Exception as e:
        if conn is not None:
            conn.rollback()
        logging.exception("POSTGRES_UTILS.PY: Database error while saving get_started response", exc_info=e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to fetch get_started data for a given auth_provider_id
def get_user_get_started_data(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        
        query_stmt = """
            SELECT *
            FROM get_started
            WHERE auth_provider_id = %s and is_active=true
            ORDER BY pk_id DESC
        """

        # Execute the SQL query
        cursor.execute(query_stmt, (auth_provider_id,))
        
        # Fetch all the rows matching the auth_provider_id
        results = cursor.fetchall()

        column_names = [desc[0] for desc in cursor.description]  # Get column names
        get_started_entries = [dict(zip(column_names, row)) for row in results]  # Convert rows to dictionaries

        return get_started_entries

    except Exception as e:
        logging.error("Database error while fetching get_started data for user: %s", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#this will be used by google cloud tasks to update behind the scenes            
def update_writing_style_record(gcp_project_id, pk_id, writing_style):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # Prepare the SQL statement to update the writing_style column
        update_stmt = """
            UPDATE get_started
            SET writing_style = %s
            WHERE pk_id = %s;
        """

        # Execute the SQL statement
        cursor.execute(update_stmt, (writing_style, pk_id,))

        # Commit the changes to the database
        conn.commit()
        logging.info('*POSTGRES_UTILS.PY: Writing style updated successfully for pk_id %s.', pk_id)
    except Exception as e:
        if conn is not None:
            # If an error occurred, rollback the transaction
            conn.rollback()
        logging.exception("POSTGRES_UTILS.PY: Database error while updating writing style", exc_info=e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_ai_created_content(gcp_project_id, gs_pk_id, new_content):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)  
        cursor = conn.cursor()

        # SQL insert statement
        insert_stmt = """
            INSERT INTO ai_created_content (gs_pk_id, new_content_payload) VALUES (%s, %s)
        """

        # Execute the SQL insert statement
        cursor.execute(insert_stmt, (gs_pk_id, new_content))
        conn.commit()
        logging.info(f"POSTGRES_UTILS.PY: New AI-generated content successfully saved for get_started PK ID: {gs_pk_id}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to get a single get_started entry by pk_id
def get_get_started_entry(gcp_project_id, pk_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # SQL query to retrieve a single get_started entry
        query_stmt = """
            SELECT pk_id, industry, content_objective, content_platform, keywords, brand_tone, writing_sample, writing_style
            FROM get_started
            WHERE pk_id = %s
        """
        
        # Execute the SQL query
        cursor.execute(query_stmt, (pk_id,))
        
        # Fetch the row that matches the pk_id
        result = cursor.fetchone()
        
        if result:
            column_names = [desc[0] for desc in cursor.description]  # Get column names
            return dict(zip(column_names, result))  # Convert the row to a dictionary
        else:
            return None

    except Exception as e:
        logging.error(f"Database error while fetching get_started data for pk_id {pk_id}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to get newly created ai-content
def fetch_ai_created_content(gcp_project_id, user_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()
        query_stmt = """
            SELECT *
            FROM ai_content
            WHERE auth_provider_id = %s
            ORDER BY pk_id DESC
        """
        cursor.execute(query_stmt, (user_id,))
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        content_entries = [dict(zip(column_names, row)) for row in results]
        return content_entries
    except Exception as e:
        logging.error(f"Database error while fetching AI created content for user_id {user_id}: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def fetch_ai_created_content_for_specific_user(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        query_stmt = """
            SELECT 
                ai_created_content_pk, industry, keywords, writing_sample, writing_style, content_objective, content_platform, 
                new_content_payload, ai_content_creation_timestamp
            FROM vw_merged_get_started_with_ai_content
            WHERE auth_provider_id = %s AND new_content_payload IS NOT NULL
            ORDER BY ai_created_content_pk DESC
        """

        cursor.execute(query_stmt, (auth_provider_id,))
        results = cursor.fetchall()

        # Get the column names to map to the dictionary keys
        column_names = [desc[0] for desc in cursor.description]
        content_entries = [dict(zip(column_names, row)) for row in results]

        return content_entries

    except Exception as e:
        logging.error(f"Database error while fetching AI created content for auth_provider_id {auth_provider_id}: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()            

def count_user_content_entries(gcp_project_id, auth_provider_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection(gcp_project_id)
        cursor = conn.cursor()

        # SQL query to count the number of user_content entries
        query_stmt = """
            SELECT COUNT(*)
            FROM user_content
            WHERE auth_provider_id = %s
        """

        # Execute the SQL query
        cursor.execute(query_stmt, (auth_provider_id,))
        result = cursor.fetchone()
        entry_count = result[0] if result else 0

        return entry_count

    except Exception as e:
        logging.error("Database error while counting user_content entries: %s", e)
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()