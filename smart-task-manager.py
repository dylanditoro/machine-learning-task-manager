# Smart Task Manager - AI-Powered Task Management System
# Features: NLP, Priority Scoring, Deadline Predictions, Analytics

from flask import Flask, request, jsonify, render_template_string
import sqlite3
import json
from datetime import datetime, timedelta
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            priority_score REAL,
            predicted_duration REAL,
            deadline TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            completed_at TEXT,
            actual_duration REAL,
            tags TEXT,
            category TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS task_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            tasks_completed INTEGER,
            avg_completion_time REAL,
            productivity_score REAL
        )
    ''')
    
    conn.commit()
    conn.close()

# NLP and ML Models
class TaskIntelligence:
    def __init__(self):
        self.priority_model = None
        self.duration_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.load_models()
    
    def extract_features_from_text(self, text):
        """Extract features from task description using NLP"""
        urgency_keywords = ['urgent', 'asap', 'emergency', 'critical', 'immediate', 'deadline']
        importance_keywords = ['important', 'crucial', 'vital', 'key', 'essential', 'priority']
        duration_keywords = {
            'quick': 0.5, 'fast': 0.5, 'brief': 0.5,
            'short': 1, 'medium': 3, 'long': 6,
            'detailed': 4, 'complex': 5, 'thorough': 4
        }
        
        text_lower = text.lower()
        
        urgency_score = sum(1 for word in urgency_keywords if word in text_lower)
        importance_score = sum(1 for word in importance_keywords if word in text_lower)
        
        estimated_duration = 2  # default
        for word, duration in duration_keywords.items():
            if word in text_lower:
                estimated_duration = duration
                break
        
        # Extract deadline information
        deadline_patterns = [
            r'by (\w+day)',
            r'in (\d+) days?',
            r'within (\d+) hours?',
            r'due (\w+day)'
        ]
        
        days_until_deadline = 7  # default
        for pattern in deadline_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if 'day' in match.group(1):
                    if 'tomorrow' in match.group(1):
                        days_until_deadline = 1
                    elif 'today' in match.group(1):
                        days_until_deadline = 0.5
                elif match.group(1).isdigit():
                    days_until_deadline = float(match.group(1))
        
        return {
            'urgency_score': urgency_score,
            'importance_score': importance_score,
            'estimated_duration': estimated_duration,
            'days_until_deadline': days_until_deadline,
            'text_length': len(text.split())
        }
    
    def calculate_priority_score(self, title, description=""):
        """Calculate intelligent priority score"""
        full_text = f"{title} {description}"
        features = self.extract_features_from_text(full_text)
        
        # Rule-based scoring with ML enhancement
        base_score = (
            features['urgency_score'] * 3 +
            features['importance_score'] * 2 +
            (10 / max(features['days_until_deadline'], 0.1)) +
            min(features['text_length'] / 10, 2)
        )
        
        # Normalize to 0-10 scale
        priority_score = min(max(base_score, 0), 10)
        
        return round(priority_score, 2)
    
    def predict_duration(self, title, description=""):
        """Predict task completion duration"""
        full_text = f"{title} {description}"
        features = self.extract_features_from_text(full_text)
        
        # Use extracted duration estimate as baseline
        predicted_duration = features['estimated_duration']
        
        # Adjust based on complexity indicators
        complexity_words = ['analyze', 'research', 'develop', 'create', 'design', 'implement']
        simple_words = ['check', 'review', 'update', 'send', 'call', 'email']
        
        text_lower = full_text.lower()
        
        if any(word in text_lower for word in complexity_words):
            predicted_duration *= 1.5
        elif any(word in text_lower for word in simple_words):
            predicted_duration *= 0.7
        
        return round(predicted_duration, 2)
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists('priority_model.pkl'):
                with open('priority_model.pkl', 'rb') as f:
                    self.priority_model = pickle.load(f)
            if os.path.exists('duration_model.pkl'):
                with open('duration_model.pkl', 'rb') as f:
                    self.duration_model = pickle.load(f)
        except:
            pass
    
    def save_models(self):
        """Save trained models"""
        try:
            if self.priority_model:
                with open('priority_model.pkl', 'wb') as f:
                    pickle.dump(self.priority_model, f)
            if self.duration_model:
                with open('duration_model.pkl', 'wb') as f:
                    pickle.dump(self.duration_model, f)
        except:
            pass

# Initialize components
task_ai = TaskIntelligence()
init_db()

# API Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    
    status = request.args.get('status', 'all')
    sort_by = request.args.get('sort', 'priority_score')
    
    if status == 'all':
        c.execute(f'SELECT * FROM tasks ORDER BY {sort_by} DESC')
    else:
        c.execute(f'SELECT * FROM tasks WHERE status = ? ORDER BY {sort_by} DESC', (status,))
    
    tasks = []
    for row in c.fetchall():
        task = {
            'id': row[0], 'title': row[1], 'description': row[2],
            'priority_score': row[3], 'predicted_duration': row[4],
            'deadline': row[5], 'status': row[6], 'created_at': row[7],
            'completed_at': row[8], 'actual_duration': row[9],
            'tags': row[10], 'category': row[11]
        }
        tasks.append(task)
    
    conn.close()
    return jsonify(tasks)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    
    title = data.get('title', '')
    description = data.get('description', '')
    category = data.get('category', 'general')
    
    # AI-powered analysis
    priority_score = task_ai.calculate_priority_score(title, description)
    predicted_duration = task_ai.predict_duration(title, description)
    
    # Auto-generate deadline if not provided
    deadline = data.get('deadline')
    if not deadline:
        days_ahead = max(1, int(predicted_duration / 8))  # Assume 8 hours per day
        deadline = (datetime.now() + timedelta(days=days_ahead)).isoformat()
    
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO tasks (title, description, priority_score, predicted_duration, 
                          deadline, created_at, tags, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, description, priority_score, predicted_duration, 
          deadline, datetime.now().isoformat(), 
          data.get('tags', ''), category))
    
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'id': task_id,
        'title': title,
        'priority_score': priority_score,
        'predicted_duration': predicted_duration,
        'deadline': deadline,
        'message': 'Task created with AI analysis'
    })

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.json
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    
    # Handle task completion
    if data.get('status') == 'completed':
        actual_duration = data.get('actual_duration', 0)
        c.execute('''
            UPDATE tasks SET status = ?, completed_at = ?, actual_duration = ?
            WHERE id = ?
        ''', ('completed', datetime.now().isoformat(), actual_duration, task_id))
    else:
        # General update
        fields = []
        values = []
        for field in ['title', 'description', 'priority_score', 'deadline', 'status', 'category']:
            if field in data:
                fields.append(f"{field} = ?")
                values.append(data[field])
        
        if fields:
            values.append(task_id)
            c.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values)
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Task updated successfully'})

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    c.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Task deleted successfully'})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    conn = sqlite3.connect('tasks.db')
    c = conn.cursor()
    
    # Overall statistics
    c.execute('SELECT COUNT(*) FROM tasks')
    total_tasks = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM tasks WHERE status = "completed"')
    completed_tasks = c.fetchone()[0]
    
    c.execute('SELECT AVG(actual_duration) FROM tasks WHERE status = "completed" AND actual_duration > 0')
    avg_completion_time = c.fetchone()[0] or 0
    
    # Priority distribution
    c.execute('SELECT priority_score, COUNT(*) FROM tasks GROUP BY CAST(priority_score AS INTEGER)')
    priority_dist = dict(c.fetchall())
    
    # Recent productivity (last 7 days)
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    c.execute('SELECT COUNT(*) FROM tasks WHERE completed_at > ? AND status = "completed"', (week_ago,))
    weekly_completed = c.fetchone()[0]
    
    # Accuracy of predictions
    c.execute('''
        SELECT AVG(ABS(predicted_duration - actual_duration)) 
        FROM tasks 
        WHERE status = "completed" AND actual_duration > 0
    ''')
    prediction_accuracy = c.fetchone()[0] or 0
    
    conn.close()
    
    productivity_score = min((weekly_completed / 7) * 10, 10)  # Scale to 0-10
    
    return jsonify({
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'completion_rate': round((completed_tasks / max(total_tasks, 1)) * 100, 1),
        'avg_completion_time': round(avg_completion_time, 2),
        'weekly_completed': weekly_completed,
        'productivity_score': round(productivity_score, 2),
        'priority_distribution': priority_dist,
        'prediction_accuracy': round(prediction_accuracy, 2)
    })

@app.route('/api/smart-suggestions', methods=['POST'])
def smart_suggestions():
    """Provide intelligent suggestions based on task analysis"""
    data = request.json
    text = data.get('text', '')
    
    suggestions = {
        'category': 'general',
        'tags': [],
        'similar_tasks': [],
        'recommendations': []
    }
    
    # Category detection
    categories = {
        'work': ['meeting', 'report', 'presentation', 'project', 'deadline', 'client'],
        'personal': ['home', 'family', 'health', 'exercise', 'shopping', 'personal'],
        'learning': ['study', 'course', 'tutorial', 'learn', 'research', 'book'],
        'maintenance': ['fix', 'repair', 'clean', 'organize', 'update', 'maintenance']
    }
    
    text_lower = text.lower()
    for cat, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            suggestions['category'] = cat
            break
    
    # Auto-generate tags
    important_words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    suggestions['tags'] = important_words[:5]  # Top 5 relevant words
    
    # Recommendations
    features = task_ai.extract_features_from_text(text)
    if features['urgency_score'] > 0:
        suggestions['recommendations'].append("High urgency detected - consider prioritizing this task")
    
    if features['estimated_duration'] > 4:
        suggestions['recommendations'].append("Complex task detected - consider breaking into smaller subtasks")
    
    if features['days_until_deadline'] < 2:
        suggestions['recommendations'].append("Tight deadline - allocate focused time blocks")
    
    return jsonify(suggestions)

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Task Manager</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { text-align: center; margin-bottom: 30px; }
        h1 { color: #2c3e50; margin-bottom: 10px; }
        .subtitle { color: #7f8c8d; }
        
        .dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        
        .main-content { display: grid; grid-template-columns: 1fr 350px; gap: 30px; }
        .task-section { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        
        .task-form { margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #2c3e50; font-weight: 500; }
        input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        textarea { resize: vertical; min-height: 80px; }
        
        .btn { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; transition: background 0.3s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #229954; }
        
        .task-list { }
        .task-item { background: #f8f9fc; border: 1px solid #e1e8ed; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .task-header { display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }
        .task-title { font-weight: bold; color: #2c3e50; flex: 1; }
        .priority-badge { padding: 4px 12px; border-radius: 20px; color: white; font-size: 12px; font-weight: bold; }
        .priority-high { background: #e74c3c; }
        .priority-medium { background: #f39c12; }
        .priority-low { background: #27ae60; }
        
        .task-meta { font-size: 12px; color: #7f8c8d; margin-bottom: 8px; }
        .task-description { color: #5a6c7d; margin-bottom: 10px; }
        .task-actions { display: flex; gap: 10px; }
        .task-actions button { padding: 6px 12px; font-size: 12px; }
        
        .analytics-panel { }
        .analytics-section { margin-bottom: 25px; }
        .analytics-title { color: #2c3e50; margin-bottom: 15px; font-weight: bold; }
        .progress-bar { background: #ecf0f1; border-radius: 10px; height: 20px; overflow: hidden; }
        .progress-fill { height: 100%; background: #3498db; transition: width 0.3s; }
        
        .filter-bar { display: flex; gap: 15px; margin-bottom: 20px; align-items: center; }
        .filter-bar select { width: auto; }
        
        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr 1fr; }
            .main-content { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Smart Task Manager</h1>
            <p class="subtitle">AI-powered task management with intelligent priority scoring and analytics</p>
        </header>

        <div class="dashboard" id="dashboard">
            <!-- Stats will be loaded here -->
        </div>

        <div class="main-content">
            <div class="task-section">
                <h2>Task Management</h2>
                
                <form class="task-form" id="taskForm">
                    <div class="form-group">
                        <label for="title">Task Title</label>
                        <input type="text" id="title" required placeholder="e.g., Complete quarterly report by Friday">
                    </div>
                    
                    <div class="form-group">
                        <label for="description">Description</label>
                        <textarea id="description" placeholder="Detailed description of the task..."></textarea>
                    </div>
                    
                    <div style="display: flex; gap: 15px;">
                        <div class="form-group" style="flex: 1;">
                            <label for="category">Category</label>
                            <select id="category">
                                <option value="general">General</option>
                                <option value="work">Work</option>
                                <option value="personal">Personal</option>
                                <option value="learning">Learning</option>
                                <option value="maintenance">Maintenance</option>
                            </select>
                        </div>
                        
                        <div class="form-group" style="flex: 1;">
                            <label for="deadline">Deadline</label>
                            <input type="datetime-local" id="deadline">
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">Create Task with AI Analysis</button>
                </form>

                <div class="filter-bar">
                    <label>Filter:</label>
                    <select id="statusFilter">
                        <option value="all">All Tasks</option>
                        <option value="pending">Pending</option>
                        <option value="in_progress">In Progress</option>
                        <option value="completed">Completed</option>
                    </select>
                    
                    <label>Sort by:</label>
                    <select id="sortBy">
                        <option value="priority_score">Priority Score</option>
                        <option value="deadline">Deadline</option>
                        <option value="created_at">Created Date</option>
                        <option value="predicted_duration">Duration</option>
                    </select>
                </div>

                <div class="task-list" id="taskList">
                    <!-- Tasks will be loaded here -->
                </div>
            </div>

            <div class="analytics-panel">
                <div class="task-section">
                    <h3>📊 Productivity Analytics</h3>
                    
                    <div class="analytics-section">
                        <div class="analytics-title">Completion Rate</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="completionProgress" style="width: 0%"></div>
                        </div>
                        <div style="text-align: center; margin-top: 5px; font-size: 12px; color: #7f8c8d;" id="completionText">0%</div>
                    </div>
                    
                    <div class="analytics-section">
                        <div class="analytics-title">Weekly Productivity</div>
                        <div id="weeklyStats">Loading...</div>
                    </div>
                    
                    <div class="analytics-section">
                        <div class="analytics-title">AI Prediction Accuracy</div>
                        <div id="accuracyStats">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let tasks = [];
        let analytics = {};

        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadTasks();
            loadAnalytics();
            
            // Set up event listeners
            document.getElementById('taskForm').addEventListener('submit', createTask);
            document.getElementById('statusFilter').addEventListener('change', loadTasks);
            document.getElementById('sortBy').addEventListener('change', loadTasks);
        });

        async function loadTasks() {
            const status = document.getElementById('statusFilter').value;
            const sort = document.getElementById('sortBy').value;
            
            const response = await fetch(`/api/tasks?status=${status}&sort=${sort}`);
            tasks = await response.json();
            renderTasks();
        }

        async function loadAnalytics() {
            const response = await fetch('/api/analytics');
            analytics = await response.json();
            renderDashboard();
            renderAnalytics();
        }

        function renderDashboard() {
            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = `
                <div class="stat-card">
                    <div class="stat-number">${analytics.total_tasks || 0}</div>
                    <div class="stat-label">Total Tasks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${analytics.completed_tasks || 0}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${analytics.productivity_score || 0}</div>
                    <div class="stat-label">Productivity Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${analytics.avg_completion_time || 0}h</div>
                    <div class="stat-label">Avg Completion Time</div>
                </div>
            `;
        }

        function renderAnalytics() {
            const completionRate = analytics.completion_rate || 0;
            document.getElementById('completionProgress').style.width = completionRate + '%';
            document.getElementById('completionText').textContent = completionRate + '%';
            
            document.getElementById('weeklyStats').innerHTML = `
                <strong>${analytics.weekly_completed || 0}</strong> tasks completed this week<br>
                <small>Productivity Score: ${analytics.productivity_score || 0}/10</small>
            `;
            
            document.getElementById('accuracyStats').innerHTML = `
                Average prediction error: <strong>${analytics.prediction_accuracy || 0}h</strong><br>
                <small>AI is getting smarter with each completed task!</small>
            `;
        }

        function renderTasks() {
            const taskList = document.getElementById('taskList');
            
            if (tasks.length === 0) {
                taskList.innerHTML = '<p style="text-align: center; color: #7f8c8d; padding: 20px;">No tasks found. Create your first task!</p>';
                return;
            }

            taskList.innerHTML = tasks.map(task => {
                const priority = task.priority_score >= 7 ? 'high' : task.priority_score >= 4 ? 'medium' : 'low';
                const deadline = task.deadline ? new Date(task.deadline).toLocaleDateString() : 'No deadline';
                
                return `
                    <div class="task-item">
                        <div class="task-header">
                            <div class="task-title">${task.title}</div>
                            <span class="priority-badge priority-${priority}">
                                Priority: ${task.priority_score}/10
                            </span>
                        </div>
                        
                        <div class="task-meta">
                            📅 ${deadline} • ⏱️ ${task.predicted_duration}h estimated • 📂 ${task.category}
                        </div>
                        
                        ${task.description ? `<div class="task-description">${task.description}</div>` : ''}
                        
                        <div class="task-actions">
                            ${task.status !== 'completed' ? `
                                <button class="btn btn-success" onclick="completeTask(${task.id})">
                                    Complete
                                </button>
                            ` : `
                                <span style="color: #27ae60; font-weight: bold;">✅ Completed</span>
                            `}
                            <button class="btn btn-danger" onclick="deleteTask(${task.id})">
                                Delete
                            </button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function createTask(e) {
            e.preventDefault();
            
            const formData = {
                title: document.getElementById('title').value,
                description: document.getElementById('description').value,
                category: document.getElementById('category').value,
                deadline: document.getElementById('deadline').value || null
            };

            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (response.ok) {
                document.getElementById('taskForm').reset();
                loadTasks();
                loadAnalytics();
                
                const result = await response.json();
                alert(`Task created! AI Analysis:\\nPriority Score: ${result.priority_score}/10\\nEstimated Duration: ${result.predicted_duration} hours`);
            }
        }

        async function completeTask(taskId) {
            const actualDuration = prompt('How many hours did this task actually take? (optional)', '');
            
            const response = await fetch(`/api/tasks/${taskId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    status: 'completed',
                    actual_duration: actualDuration ? parseFloat(actualDuration) : 0
                })
            });

            if (response.ok) {
                loadTasks();
                loadAnalytics();
            }
        }

        async function deleteTask(taskId) {
            if (confirm('Are you sure you want to delete this task?')) {
                const response = await fetch(`/api/tasks/${taskId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    loadTasks();
                    loadAnalytics();
                }
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Starting Smart Task Manager...")
    print("Features enabled:")
    print("  • Natural Language Processing for task analysis")
    print("  • Intelligent priority scoring")
    print("  • Duration prediction")
    print("  • Productivity analytics")
    print("  • Real-time dashboard")
    print("\nAccess the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

