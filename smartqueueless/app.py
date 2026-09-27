from datetime import datetime
from functools import wraps
import os

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-change-this"
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///smartqueueless.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# ============================================================
# DATABASE MODELS
# ============================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user", nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    tokens = db.relationship(
        "Token",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Service(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(150), nullable=False)
    prefix = db.Column(db.String(10), nullable=False)
    average_service_time = db.Column(db.Float, default=5.0)
    active = db.Column(db.Boolean, default=True)

    tokens = db.relationship(
        "Token",
        backref="service",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Token(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    token_number = db.Column(db.Integer, nullable=False)
    token_code = db.Column(db.String(30), nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    service_id = db.Column(db.Integer, db.ForeignKey("service.id"), nullable=False)

    status = db.Column(
        db.String(20),
        default="waiting",
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    called_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ============================================================
# WAITING-TIME PREDICTION
# ============================================================

def predict_waiting_time(service_id, token_id=None):
    """
    Predict waiting time using:
    1. Number of people ahead
    2. Currently serving token
    3. Historical completed service durations
    4. Current time of day

    Falls back to the service's average time when there
    is not enough historical data for ML.
    """

    service = db.session.get(Service, service_id)

    if not service:
        return 0

    # Find the user's position in the queue
    if token_id:
        current_token = db.session.get(Token, token_id)

        if current_token:
            people_ahead = Token.query.filter(
                Token.service_id == service_id,
                Token.status == "waiting",
                Token.token_number < current_token.token_number
            ).count()
        else:
            people_ahead = Token.query.filter_by(
                service_id=service_id,
                status="waiting"
            ).count()
    else:
        people_ahead = Token.query.filter_by(
            service_id=service_id,
            status="waiting"
        ).count()

    # Check whether somebody is currently being served
    active_token = Token.query.filter_by(
        service_id=service_id,
        status="serving"
    ).first()

    active_count = 1 if active_token else 0

    # ---------------------------------------------------------
    # Get historical completed tokens
    # ---------------------------------------------------------

    historical_tokens = Token.query.filter(
        Token.service_id == service_id,
        Token.status == "completed",
        Token.called_at.isnot(None),
        Token.completed_at.isnot(None)
    ).order_by(
        Token.completed_at.desc()
    ).limit(100).all()

    historical_data = []

    for token in historical_tokens:
        duration = (
            token.completed_at - token.called_at
        ).total_seconds() / 60

        if duration > 0:
            historical_data.append({
                "token_number": token.token_number,
                "hour": token.called_at.hour,
                "duration": duration
            })

    # ---------------------------------------------------------
    # Machine Learning prediction
    # ---------------------------------------------------------

    if len(historical_data) >= 5:
        X = []
        y = []

        for item in historical_data:
            X.append([
                item["token_number"],
                item["hour"]
            ])
            y.append(item["duration"])

        model = LinearRegression()

        model.fit(
            np.array(X),
            np.array(y)
        )

        current_hour = datetime.now().hour

        predicted_service_time = float(
            model.predict(
                np.array([
                    [
                        people_ahead + 1,
                        current_hour
                    ]
                ])
            )[0]
        )

        # Keep prediction within a practical range
        predicted_service_time = max(
            2.0,
            min(predicted_service_time, 60.0)
        )

    else:
        # Not enough historical data yet
        predicted_service_time = (
            service.average_service_time or 5
        )

    # ---------------------------------------------------------
    # Calculate total waiting time
    # ---------------------------------------------------------

    waiting_time = (
        people_ahead + active_count
    ) * predicted_service_time

    return max(0, round(waiting_time))


# ============================================================
# ADMIN DECORATOR
# ============================================================

def admin_required(function):
    @wraps(function)
    @login_required
    def wrapper(*args, **kwargs):
        if current_user.role != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))

        return function(*args, **kwargs)

    return wrapper


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    services = Service.query.filter_by(active=True).all()

    return render_template(
        "index.html",
        services=services
    )


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash(
                "Password must contain at least 8 characters.",
                "error"
            )
            return redirect(url_for("register"))

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash("Email already registered.", "error")
            return redirect(url_for("register"))

        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            role="user"
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(
            user.password,
            password
        ):
            login_user(user)

            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("index"))


# ============================================================
# USER DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    services = Service.query.filter_by(
        active=True
    ).all()

    user_tokens = Token.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Token.created_at.desc()
    ).all()

    return render_template(
        "dashboard.html",
        services=services,
        tokens=user_tokens
    )


# ============================================================
# GENERATE DIGITAL TOKEN
# ============================================================

@app.route("/join-queue/<int:service_id>", methods=["POST"])
@login_required
def join_queue(service_id):

    service = db.session.get(Service, service_id)

    if not service or not service.active:
        flash(
            "This service is currently unavailable.",
            "error"
        )
        return redirect(url_for("dashboard"))

    existing_token = Token.query.filter(
        Token.user_id == current_user.id,
        Token.service_id == service_id,
        Token.status.in_(["waiting", "serving"])
    ).first()

    if existing_token:
        flash(
            "You already have an active token for this service.",
            "error"
        )
        return redirect(url_for("queue_status", token_id=existing_token.id))

    latest_token = Token.query.filter_by(
        service_id=service_id
    ).order_by(
        Token.token_number.desc()
    ).first()

    next_number = (
        latest_token.token_number + 1
        if latest_token
        else 1
    )

    token = Token(
        token_number=next_number,
        token_code=f"{service.prefix}{next_number:03d}",
        user_id=current_user.id,
        service_id=service_id,
        status="waiting"
    )

    db.session.add(token)
    db.session.commit()

    flash(
        f"Token {token.token_code} generated successfully.",
        "success"
    )

    return redirect(
        url_for(
            "queue_status",
            token_id=token.id
        )
    )


# ============================================================
# QUEUE STATUS
# ============================================================

@app.route("/queue/<int:token_id>")
@login_required
def queue_status(token_id):

    token = db.session.get(Token, token_id)

    if not token:
        flash("Token not found.", "error")
        return redirect(url_for("dashboard"))

    # Only token owner or admin can view the queue
    if token.user_id != current_user.id and current_user.role != "admin":
        flash("Unauthorized access.", "error")
        return redirect(url_for("dashboard"))

    # People waiting before this token
    people_ahead = Token.query.filter(
        Token.service_id == token.service_id,
        Token.status == "waiting",
        Token.token_number < token.token_number
    ).count()

    # Find currently serving token
    current_serving = Token.query.filter_by(
        service_id=token.service_id,
        status="serving"
    ).order_by(
        Token.called_at.asc()
    ).first()

    # Queue position
    if token.status == "waiting":
        queue_position = people_ahead + 1
    else:
        queue_position = 0

    # Waiting-time prediction
    if token.status == "waiting":
        waiting_time = predict_waiting_time(
            token.service_id,
            token.id
        )
    else:
        waiting_time = 0

    return render_template(
        "queue.html",
        token=token,
        people_ahead=people_ahead,
        queue_position=queue_position,
        waiting_time=waiting_time,
        current_serving=current_serving
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    services = Service.query.all()

    # FIFO Update: Fetch tokens ordered by oldest created first (asc)
    tokens = Token.query.order_by(
        Token.created_at.asc()
    ).limit(100).all()

    # Find the next token ID in line for each service
    next_token_ids = []
    for service in services:
        first_waiting = Token.query.filter_by(
            service_id=service.id,
            status="waiting"
        ).order_by(Token.created_at.asc()).first()
        
        if first_waiting:
            next_token_ids.append(first_waiting.id)

    waiting_count = Token.query.filter_by(
        status="waiting"
    ).count()

    serving_count = Token.query.filter_by(
        status="serving"
    ).count()

    completed_count = Token.query.filter_by(
        status="completed"
    ).count()

    return render_template(
        "admin.html",
        services=services,
        tokens=tokens,
        next_token_ids=next_token_ids,
        waiting_count=waiting_count,
        serving_count=serving_count,
        completed_count=completed_count
    )


# ============================================================
# ADMIN - CREATE SERVICE
# ============================================================

@app.route("/admin/service/create", methods=["POST"])
@admin_required
def create_service():

    name = request.form.get("name", "").strip()
    location = request.form.get("location", "").strip()
    prefix = request.form.get("prefix", "").strip().upper()

    average_time = request.form.get(
        "average_service_time",
        "5"
    )

    try:
        average_time = float(average_time)
    except ValueError:
        average_time = 5.0

    if not name or not location or not prefix:
        flash(
            "Service name, location and prefix are required.",
            "error"
        )
        return redirect(url_for("admin_dashboard"))

    service = Service(
        name=name,
        location=location,
        prefix=prefix,
        average_service_time=max(1, average_time),
        active=True
    )

    db.session.add(service)
    db.session.commit()

    flash(
        "Service created successfully.",
        "success"
    )

    return redirect(url_for("admin_dashboard"))


# ============================================================
# ADMIN - CALL NEXT TOKEN
# ============================================================

@app.route("/admin/token/<int:token_id>/serve", methods=["POST"])
@admin_required
def serve_token(token_id):

    token = db.session.get(Token, token_id)

    if not token:
        flash("Token not found.", "error")
        return redirect(url_for("admin_dashboard"))

    # FIFO Check: Verify target token is the oldest waiting token for this service
    first_in_line = Token.query.filter_by(
        service_id=token.service_id,
        status="waiting"
    ).order_by(Token.created_at.asc()).first()

    if first_in_line and first_in_line.id != token.id:
        flash(
            "Cannot serve out of order. Please serve the first person in line for this service.",
            "error"
        )
        return redirect(url_for("admin_dashboard"))

    currently_serving = Token.query.filter(
        Token.service_id == token.service_id,
        Token.status == "serving"
    ).first()

    if currently_serving:
        flash(
            "A token is already being served for this service.",
            "error"
        )
        return redirect(url_for("admin_dashboard"))

    token.status = "serving"
    token.called_at = datetime.utcnow()

    db.session.commit()

    flash(
        f"{token.token_code} is now being served.",
        "success"
    )

    return redirect(url_for("admin_dashboard"))


# ============================================================
# ADMIN - COMPLETE TOKEN
# ============================================================

@app.route("/admin/token/<int:token_id>/complete", methods=["POST"])
@admin_required
def complete_token(token_id):

    token = db.session.get(Token, token_id)

    if not token:
        flash("Token not found.", "error")
        return redirect(url_for("admin_dashboard"))

    token.status = "completed"
    token.completed_at = datetime.utcnow()

    db.session.commit()

    flash(
        f"{token.token_code} completed.",
        "success"
    )

    return redirect(url_for("admin_dashboard"))


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    db.create_all()

    # Create admin account if it doesn't exist.
    admin_email = os.environ.get(
        "ADMIN_EMAIL",
        "admin@smartqueueless.com"
    )

    admin_password = os.environ.get(
        "ADMIN_PASSWORD",
        "Admin@123"
    )

    admin = User.query.filter_by(
        email=admin_email
    ).first()

    if not admin:
        admin = User(
            name="System Administrator",
            email=admin_email,
            password=generate_password_hash(admin_password),
            role="admin"
        )
        db.session.add(admin)

    # Create initial services.
    if Service.query.count() == 0:

        services = [
            Service(
                name="Hospital Registration",
                location="Main Hospital",
                prefix="H",
                average_service_time=8
            ),
            Service(
                name="Bank Customer Service",
                location="Main Branch",
                prefix="B",
                average_service_time=6
            ),
            Service(
                name="College Office",
                location="College Administration",
                prefix="C",
                average_service_time=5
            ),
            Service(
                name="Service Center",
                location="Customer Service Center",
                prefix="S",
                average_service_time=7
            )
        ]

        db.session.add_all(services)

    db.session.commit()


# ============================================================
# START APPLICATION
# ============================================================

with app.app_context():
    initialize_database()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )

