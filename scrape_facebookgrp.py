import requests
import json

# =========================
# 1️⃣ CONFIGURATION
# =========================

ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"  # Replace with your access token
GROUP_ID = "706953679342592"  # Facebook group ID

# Output file
output_file = "facebook_posts.txt"

# =========================
# 2️⃣ FETCH POSTS WITH PAGINATION
# =========================


def fetch_group_posts(group_id, access_token, limit=50):
    """
    Fetch all posts from a Facebook group using Graph API with pagination.
    """
    all_posts = []
    url = f"https://graph.facebook.com/v18.0/{group_id}/feed"
    params = {
        "fields": "message,created_time,from",
        "access_token": access_token,
        "limit": limit,
    }

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("Error fetching data:", response.text)
            break

        data = response.json()
        posts = data.get("data", [])
        all_posts.extend(posts)

        # Pagination: move to next page if available
        paging = data.get("paging", {})
        next_page = paging.get("next")
        if next_page:
            url = next_page
            params = {}  # no need for params when using next
        else:
            break

    return all_posts


# =========================
# 3️⃣ SAVE POSTS TO TXT
# =========================


def save_posts_to_txt(posts, file_path):
    """
    Save posts messages to a txt file, separated by two newlines.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for post in posts:
            if "message" in post:
                message = post["message"]
                created_time = post.get("created_time", "")
                author = post.get("from", {}).get("name", "")
                f.write(f"{author} ({created_time}):\n{message}\n\n")
    print(f"Saved {len(posts)} posts to {file_path}")


# =========================
# 4️⃣ EXECUTE
# =========================

print("Fetching posts from Facebook group...")
posts = fetch_group_posts(GROUP_ID, ACCESS_TOKEN, limit=50)
print(f"Total posts fetched: {len(posts)}")

save_posts_to_txt(posts, output_file)
