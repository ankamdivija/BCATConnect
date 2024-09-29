from flask import Flask, jsonify,render_template, request
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pytz  # Import for timezone handling
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

employees = [
   {
       'name': 'Alice Smith',
       'skills': ['Art Project Management', 'Creative Collaboration', 'Documentation'],
       'project': 'Art Exhibits Management',
       'desk_location': 'Desk 101',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img1.jpeg'
   },
   {
       'name': 'Mia King',
       'skills': ['Team Collaboration', 'Community Outreach', 'Event Coordination'],
       'project': 'Community Arts Program',
       'desk_location': 'Desk 113',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/image.jpg'
   },
   {
       'name': 'John Doe',
       'skills': ['Art Event Coordination', 'Public Speaking', 'Creative Workshops'],
       'project': 'Collaborative Art Workshops',
       'desk_location': 'Desk 102',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/img2.jpeg'
   },
   {
       'name': 'Mary Jane',
       'skills': ['Cultural Awareness', 'Artist Coaching and Support', 'Creative Training'],
       'project': 'Artist Mentorship Program',
       'desk_location': 'Desk 103',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img3.jpeg'
   },
   {
       'name': 'Robert Brown',
       'skills': ['Artistic Tool Integration', 'Collaborative Design', 'Art Technology Training'],
       'project': 'Digital Art Tools Training',
       'desk_location': 'Desk 104',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/img1.jpeg'
   },
   {
       'name': 'Linda Green',
       'skills': ['Web Design', 'User Experience Design for Artists', 'Frontend Development'],
       'project': 'Artist Portfolio Website',
       'desk_location': 'Desk 105',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img2.jpeg'
   },
   {
       'name': 'James Wilson',
       'skills': ['Creative Event Planning', 'Art Exhibits', 'Team Collaboration'],
       'project': 'Art Team Bonding Activities',
       'desk_location': 'Desk 106',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img3.jpeg'
   },
   {
       'name': 'Sophia White',
       'skills': ['Art Data Analysis', 'Artist Interaction Metrics', 'Gallery Management'],
       'project': 'Art Team Interaction Tracking',
       'desk_location': 'Desk 107',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/img1.jpeg'
   },
   {
       'name': 'Michael Johnson',
       'skills': ['Conflict Resolution in Creative Teams', 'Art Team Building', 'Communication in Collaborative Projects'],
       'project': 'Cross-Disciplinary Art Engagement',
       'desk_location': 'Desk 108',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img2.jpeg'
   },
   {
       'name': 'Emily Davis',
       'skills': ['Artistic App Development', 'Mobile User Experience for Art', 'Art Cross-Platform Integration'],
       'project': 'Art Collaboration Mobile App',
       'desk_location': 'Desk 109',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/img3.jpeg'
   },
   {
       'name': 'William Martinez',
       'skills': ['Creative AI Tools', 'Artistic Machine Learning', 'Interactive Art Systems'],
       'project': 'AI-Powered Art Collaboration',
       'desk_location': 'Desk 110',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img1.jpeg'
   },
   {
       'name': 'Olivia Taylor',
       'skills': ['Backend Systems for Art Platforms', 'Cloud Infrastructure for Art Collaboration', 'Art Management Systems'],
       'project': 'Backend for BCAT Art Projects',
       'desk_location': 'Desk 111',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/img2.jpeg'
   },
   {
       'name': 'Liam Anderson',
       'skills': ['Python for Art Projects', 'API Development for Art Platforms', 'Collaborative Frameworks'],
       'project': 'Art Collaboration Backend',
       'desk_location': 'Desk 112',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/img3.jpeg'
   },
   {
       'name': 'Noah Harris',
       'skills': ['Art Data Management', 'SQL for Art Platforms', 'Creative Data Warehousing'],
       'project': 'Artist Data Management System',
       'desk_location': 'Desk 114',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/image.jpg'
   },
   {
       'name': 'Ava Clark',
       'skills': ['JavaScript for Art Applications', 'Frontend Development for Galleries', 'Interactive Art Platforms'],
       'project': 'Art Collaboration Dashboard',
       'desk_location': 'Desk 115',
       'availability': 'In Office',
       'available': True,
       'image_url': '/static/images/image.jpg'
   },
   {
       'name': 'Elijah Lewis',
       'skills': ['Creative Data Engineering', 'Automated Art Workflows', 'Data Pipelines for Collaboration'],
       'project': 'Art Team Interaction Data Pipeline',
       'desk_location': 'Desk 116',
       'availability': 'Remote',
       'available': False,
       'image_url': '/static/images/image.jpg'
   }
]

# Combine employee details into a single text string for embedding
def create_employee_text(employee):
    # Giving more weight to skills by repeating them
    skills_text = " ".join(employee['skills']) * 2
    combined_text = f"{employee['name']} {skills_text} {employee['project']} {employee['desk_location']} {employee['availability']}"
    return combined_text

# Encode employee details to vectors using the pre-trained model
employee_texts = [create_employee_text(emp) for emp in employees]
employee_embeddings = model.encode(employee_texts, convert_to_tensor=True)

@app.route('/')
def homepage() :
    return render_template("index.html")

@app.route('/employees')
def directory():
    return render_template('directory.html', employees=employees)

@app.route('/knowmore')
def contact_us():
    return render_template('contact_us.html')

@app.route('/search', methods=['GET'])
def search():
    original_query = request.args.get('query', '')
    query = request.args.get('query', '').strip().lower()  # Preprocess the query

    # Encode the user query using the pre-trained model
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and employee details
    similarities = util.cos_sim(query_embedding, employee_embeddings)[0]
    similarity_scores = similarities.cpu().numpy()

    # Set a similarity threshold to filter relevant results (e.g., 0.3)
    threshold = 0.3
    filtered_indices = np.where(similarity_scores >= threshold)[0]

    # Sort the filtered indices based on similarity scores in descending order
    filtered_indices = filtered_indices[np.argsort(similarity_scores[filtered_indices])[::-1]]

    # Get the filtered matching employees
    filtered_employees = [employees[idx] for idx in filtered_indices]

    # Render the template with filtered employees
    return render_template('directory.html', employees=filtered_employees, searchquery=original_query)

# Example project data with team members and contact information
# projects = [
#     {
#         "description": "Developed a machine learning model for predicting customer churn using Python, Pandas, and Scikit-learn.",
#         "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn"],
#         "tools": ["Jupyter Notebook", "Git"],
#         "team_members": ["alice@example.com", "bob@example.com"]
#     },
#     {
#         "description": "Built a recommendation system for products using collaborative filtering and neural networks.",
#         "skills": ["Python", "Deep Learning", "Collaborative Filtering"],
#         "tools": ["TensorFlow", "Keras"],
#         "team_members": ["charlie@example.com", "dave@example.com"]
#     },
#     {
#         "description": "Created a data analysis pipeline for real-time sales data processing using Python and SQL.",
#         "skills": ["Python", "Data Analysis", "SQL"],
#         "tools": ["Pandas", "SQLAlchemy"],
#         "team_members": ["eve@example.com", "frank@example.com"]
#     }
# ]


projects = [
   {
       "description": "A creative after-school program teaching high school students video editing, graphic design, and audio production. Students work on real-world projects like creating promotional content for local businesses, helping them build a portfolio for college or careers.",
       "skills": ["Video Editing", "Graphic Design", "Audio Production","Storytelling", "Teamwork", "Collaboration", "Entrepreneurship", "Problem-Solving"],
       "tools": ["GitHub", "Adobe Creative Suite", "Canva", "Audacity", "Instagram", "Facebook"],
       "team_members": ["alice@example.com", "bob@example.com"]
   },
   {
       "description": "A program empowering adults to turn their artistic skills into small businesses. Participants will learn painting, sculpture, and mixed media while gaining business skills like pricing, marketing, and selling their art both online and at local markets.",
       "skills": ["Visual Arts", "Painting", "Mixed Media", "Video Editing", "Entrepreneurship", "Marketing", "Problem-Solving"],
       "tools": ["Canva", "Etsy", "Instagram", "GitHub", "MongoDB"],
       "team_members": ["eve@example.com", "frank@example.com"]
   },
   {
       "description": "An intensive bootcamp teaching adults full-stack web development. Participants learn HTML, CSS, JavaScript, React, and Node.js, building websites for real clients to gain experience and secure jobs in tech.",
       "skills": ["Web Development", "Front-End/Back-End Coding", "Technology"],
       "tools": ["Visual Studio Code", "GitHub", "React", "Node.js", "MongoDB"],
       "team_members": ["lexi@example.com", "dave@example.com"]
   }
]

# Encode the project descriptions to get embeddings
project_descriptions = [project["description"] for project in projects]
project_embeddings = model.encode(project_descriptions)

@app.route('/projects')
def index():
    return render_template('project.html', projects=projects)


@app.route('/suggest-collaborations', methods=['GET'])
def suggest_collaborations():
    # Compute cosine similarity matrix
    cosine_similarity = util.cos_sim(project_embeddings, project_embeddings)

    # Focus on a specific project (e.g., Project 1)
    project_index = 0  # Index for Project 1
    suggestions = []
    for j, similarity in enumerate(cosine_similarity[project_index]):
        if j != project_index:  # Skip the similarity with itself
            suggestion = {
                'project_1': projects[project_index]['description'],
                'project_2': projects[j]['description'],
                'similarity_score': round(similarity.item(), 4),
                'suggested_skills': projects[j]['skills'],
                'suggested_tools': projects[j]['tools']
            }
            suggestions.append(suggestion)

    # Finding the most relevant project for the focused project
    most_relevant_index = cosine_similarity[project_index][1:].argmax() + 1  # Skip the first project (itself)
    most_relevant_project = {
        'project_1': projects[project_index]['description'],
        'most_relevant_project': projects[most_relevant_index]['description'],
        'suggested_skills': projects[most_relevant_index]['skills'],
        'suggested_tools': projects[most_relevant_index]['tools'],
        'team' : projects[most_relevant_index]['team_members']
    }

    # Include the most relevant project information
    response = {
        'suggestions': suggestions,
        'most_relevant_project': most_relevant_project
    }

    return jsonify(response)


# Define working hours
WORK_START = 9  # 9 AM
WORK_END = 17   # 5 PM

# Define the range for the upcoming week (29th Sept to 5th Oct) and ensure UTC timezone
UTC = pytz.UTC  # Set timezone to UTC
START_DATE = datetime(2024, 9, 29, tzinfo=UTC)
END_DATE = datetime(2024, 10, 5, tzinfo=UTC)

# Initialize Hugging Face text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Load CSV files for all users
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Parse CSV and extract busy times
def parse_busy_times(data):
    busy_times = []
    for index, row in data.iterrows():
        start_time = pd.to_datetime(row['dtstart']).tz_convert(UTC)
        end_time = pd.to_datetime(row['dtend']).tz_convert(UTC)
        if START_DATE <= start_time <= END_DATE:
            busy_times.append((start_time, end_time))
    return busy_times

# Find free slots between working hours for a single day
def find_free_slots_for_day(busy_times, day_start, day_end):
    free_slots = []
    busy_times = sorted(busy_times)
    last_end = day_start

    for start, end in busy_times:
        if start > last_end:
            free_slots.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < day_end:
        free_slots.append((last_end, day_end))

    return free_slots


def find_common_free_slots(*user_busy_times, user_names):
    common_free_slots = []

    for day in pd.date_range(START_DATE, END_DATE):
        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=UTC) + timedelta(hours=WORK_START)
        day_end = datetime.combine(day, datetime.min.time()).replace(tzinfo=UTC) + timedelta(hours=WORK_END)

        # Find free slots for each user and keep track of their names
        user_free_slots = [
            (find_free_slots_for_day(user_times, day_start, day_end), user_name) 
            for user_times, user_name in zip(user_busy_times, user_names)
        ]

        # Find intersections between free slots of all users
        for slot_combination in zip(*[slots[0] for slots in user_free_slots]):
            latest_start = max(slot[0] for slot in slot_combination)
            earliest_end = min(slot[1] for slot in slot_combination)
            if latest_start < earliest_end:
                # Find users who are available during this slot and convert to a tuple
                available_users = tuple([slots[1] for slots, times in zip(user_free_slots, slot_combination) if latest_start >= times[0] and earliest_end <= times[1]])
                common_free_slots.append((latest_start, earliest_end, available_users))

    return common_free_slots


@app.route('/find_free_time', methods=['GET'])
def find_free_time():
    # Load CSV data from uploaded files for all users
    user1_data = load_csv_data('resources/knthara.20@gmail.com.csv')
    user2_data = load_csv_data('resources/roshini.089.cse4b@gmail.com.csv')
    user3_data = load_csv_data('resources/roshini.rk89@gmail.com.csv')

    # Extract busy times for all users
    user1_busy_times = parse_busy_times(user1_data)
    user2_busy_times = parse_busy_times(user2_data)
    user3_busy_times = parse_busy_times(user3_data)

    # Find common free slots for all three users
    user_names = ['Alice', 'Frank', 'Chris']  # Adjust these names as needed
    common_free_slots = find_common_free_slots(user1_busy_times, user2_busy_times, user3_busy_times, user_names=user_names)

    # Remove duplicate slots (now works because `available_users` is a tuple)
    unique_free_slots = list(set(common_free_slots))  # Convert to a set and back to list to remove duplicates
    unique_free_slots.sort()  # Sort the unique free slots

    # Organize free slots by day
    week_slots = {}
    for day in pd.date_range(START_DATE, END_DATE):
        day_slots = [(start, end, users) for start, end, users in unique_free_slots if start.date() == day.date()]
        week_slots[day] = day_slots

    # Pass the days of the week and week_slots to the result.html template for the calendar view
    week_days = list(pd.date_range(START_DATE, END_DATE))  # List of days for the header

    return render_template('result.html', week_slots=week_slots, week_days=week_days)


if __name__ == '__main__':
    app.run(debug=True)