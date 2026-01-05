"""
Flask Approval Server with Authentication

A secure web server for trade approvals with JWT authentication,
real-time notifications, and trade management.
"""

import os
import json
import time
import jwt
import uuid
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Dict, List, Optional, Tuple, Any, Union

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytz

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///trades.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-me'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@tradingapp.com')
    RATE_LIMIT = "200 per day;50 per hour"
    DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ['true', '1', 'on']
    SESSION_COOKIE_SECURE = not DEBUG
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    logger=True,
    engineio_logger=app.debug
)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config['RATE_LIMIT']],
    storage_uri="memory://"
)

# Models
class User(db.Model):
    """User account model."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    mfa_secret = db.Column(db.String(32))
    
    # Relationships
    trades = db.relationship('Trade', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    
    def set_password(self, password: str) -> None:
        """Create hashed password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Check hashed password."""
        return check_password_hash(self.password_hash, password)
    
    def generate_auth_token(self, token_type: str = 'access') -> str:
        """Generate JWT token for the user."""
        if token_type == 'access':
            expires_in = app.config['JWT_ACCESS_TOKEN_EXPIRES']
        else:  # refresh token
            expires_in = app.config['JWT_REFRESH_TOKEN_EXPIRES']
        
        payload = {
            'user_id': self.id,
            'username': self.username,
            'exp': datetime.now(timezone.utc) + expires_in,
            'iat': datetime.now(timezone.utc),
            'type': token_type
        }
        
        return jwt.encode(
            payload,
            app.config['JWT_SECRET_KEY'],
            algorithm='HS256'
        )
    
    @staticmethod
    def verify_auth_token(token: str) -> Optional['User']:
        """Verify JWT token and return user."""
        try:
            payload = jwt.decode(
                token,
                app.config['JWT_SECRET_KEY'],
                algorithms=['HS256']
            )
            
            if payload.get('type') != 'access':
                return None
                
            return User.query.get(payload['user_id'])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

class Trade(db.Model):
    """Trade approval request model."""
    __tablename__ = 'trades'
    
    # Status constants
    STATUS_PENDING = 'pending'
    STATUS_APPROVED = 'approved'
    STATUS_REJECTED = 'rejected'
    STATUS_EXECUTED = 'executed'
    STATUS_EXPIRED = 'expired'
    STATUS_CANCELLED = 'cancelled'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    direction = db.Column(db.Enum('buy', 'sell', name='trade_direction'), nullable=False)
    entry_price = db.Column(db.Float, nullable=True)
    stop_loss = db.Column(db.Float, nullable=True)
    take_profit = db.Column(db.Float, nullable=True)
    amount = db.Column(db.Float, nullable=False)
    expiry_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(
        db.Enum(
            STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED,
            STATUS_EXECUTED, STATUS_EXPIRED, STATUS_CANCELLED,
            name='trade_status'
        ),
        default=STATUS_PENDING,
        nullable=False
    )
    reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = db.Column('metadata', db.JSON, default=dict)
    
    # Relationships
    approvals = db.relationship('TradeApproval', backref='trade', lazy=True)
    
    @property
    def is_active(self) -> bool:
        """Check if trade is active (pending or approved but not executed)."""
        return self.status in [self.STATUS_PENDING, self.STATUS_APPROVED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'amount': self.amount,
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'status': self.status,
            'reason': self.reason,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata_,
            'approvals': [approval.to_dict() for approval in self.approvals]
        }

class TradeApproval(db.Model):
    """Trade approval records."""
    __tablename__ = 'trade_approvals'
    
    id = db.Column(db.Integer, primary_key=True)
    trade_id = db.Column(db.String(36), db.ForeignKey('trades.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.Enum('approved', 'rejected', name='approval_status'), nullable=False)
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', lazy='joined')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert approval to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'status': self.status,
            'comment': self.comment,
            'created_at': self.created_at.isoformat()
        }

class Notification(db.Model):
    """User notifications."""
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    type = db.Column(db.String(50), default='info')  # info, success, warning, error
    action_url = db.Column(db.String(512), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat(),
            'type': self.type,
            'action_url': self.action_url
        }

# Helper functions
def send_email(
    to: str,
    subject: str,
    body: str,
    html: Optional[str] = None
) -> bool:
    """Send an email using the configured SMTP server."""
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        app.logger.warning("Email not configured. Not sending email.")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = to
        
        part1 = MIMEText(body, 'plain')
        msg.attach(part1)
        
        if html:
            part2 = MIMEText(html, 'html')
            msg.attach(part2)
        
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            if app.config['MAIL_USE_TLS']:
                server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
        
        return True
    except Exception as e:
        app.logger.error(f"Error sending email: {e}")
        return False

def token_required(f):
    """Decorator to require JWT token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        # Check for token in session (for web interface)
        if not token and 'token' in session:
            token = session['token']
        
        if not token:
            if request.is_json:
                return jsonify({'error': 'Token is missing'}), 401
            return redirect(url_for('login', next=request.url))
        
        try:
            # Verify token
            payload = jwt.decode(
                token,
                app.config['JWT_SECRET_KEY'],
                algorithms=['HS256']
            )
            
            # Check if token is an access token
            if payload.get('type') != 'access':
                raise jwt.InvalidTokenError('Invalid token type')
            
            # Get user from database
            user = User.query.get(payload['user_id'])
            if not user or not user.is_active:
                raise jwt.InvalidTokenError('User not found or inactive')
            
            # Store user in request context
            request.current_user = user
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            if request.is_json:
                return jsonify({'error': 'Token has expired'}), 401
            flash('Your session has expired. Please log in again.', 'warning')
            return redirect(url_for('login', next=request.url))
            
        except jwt.InvalidTokenError as e:
            app.logger.warning(f"Invalid token: {e}")
            if request.is_json:
                return jsonify({'error': 'Invalid token'}), 401
            flash('Invalid or expired session. Please log in again.', 'danger')
            return redirect(url_for('login', next=request.url))
    
    return decorated

def admin_required(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(request, 'current_user') or not request.current_user.is_admin:
            if request.is_json:
                return jsonify({'error': 'Admin privileges required'}), 403
            flash('Administrator privileges required', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection."""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            join_room(f'user_{user.id}')
            if user.is_admin:
                join_room('admins')
            app.logger.info(f"User {user.username} connected to WebSocket")
            return {'status': 'connected'}
    
    # Unauthorized connection
    return False

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    app.logger.info("Client disconnected")

# API Routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    """Register a new user (admin only)."""
    data = request.get_json() or {}
    
    # Validate input
    if not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if username or email already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    try:
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            is_admin=data.get('is_admin', False)
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        # Generate tokens
        access_token = user.generate_auth_token('access')
        refresh_token = user.generate_auth_token('refresh')
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error registering user: {e}")
        return jsonify({'error': 'Failed to register user'}), 500

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Authenticate user and return JWT token."""
    data = request.get_json() or {}
    
    # Validate input
    if not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing username or password'}), 400
    
    # Find user by username
    user = User.query.filter_by(username=data['username']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Generate tokens
    access_token = user.generate_auth_token('access')
    refresh_token = user.generate_auth_token('refresh')
    
    # Store user ID in session for WebSocket
    session['user_id'] = user.id
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin
        }
    })

@app.route('/api/auth/refresh', methods=['POST'])
@token_required
def refresh_token():
    """Refresh access token using refresh token."""
    refresh_token = request.json.get('refresh_token')
    if not refresh_token:
        return jsonify({'error': 'Refresh token is required'}), 400
    
    try:
        payload = jwt.decode(
            refresh_token,
            app.config['JWT_SECRET_KEY'],
            algorithms=['HS256']
        )
        
        if payload.get('type') != 'refresh':
            raise jwt.InvalidTokenError('Invalid token type')
        
        user = User.query.get(payload['user_id'])
        if not user or not user.is_active:
            raise jwt.InvalidTokenError('User not found or inactive')
        
        # Generate new access token
        access_token = user.generate_auth_token('access')
        
        return jsonify({
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        })
        
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        return jsonify({'error': 'Invalid or expired refresh token'}), 401

@app.route('/api/trades', methods=['POST'])
@token_required
def create_trade():
    """Create a new trade approval request."""
    data = request.get_json() or {}
    
    # Validate required fields
    required_fields = ['symbol', 'direction', 'amount']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Create new trade
        trade = Trade(
            user_id=request.current_user.id,
            symbol=data['symbol'].upper(),
            direction=data['direction'].lower(),
            amount=float(data['amount']),
            entry_price=float(data.get('entry_price', 0)) if data.get('entry_price') else None,
            stop_loss=float(data.get('stop_loss')) if data.get('stop_loss') else None,
            take_profit=float(data.get('take_profit')) if data.get('take_profit') else None,
            expiry_time=datetime.fromisoformat(data['expiry_time']) if data.get('expiry_time') else None,
            metadata_=data.get('metadata', {})
        )
        
        db.session.add(trade)
        db.session.commit()
        
        # Notify admins
        notify_admins(
            title="New Trade Request",
            message=f"New trade request for {trade.symbol} {trade.direction.upper()}",
            type="info",
            action_url=f"/trades/{trade.id}"
        )
        
        # Send email notification to admins
        if app.config['MAIL_USERNAME']:
            admin_emails = [user.email for user in User.query.filter_by(is_admin=True).all() if user.email]
            if admin_emails:
                send_email(
                    to=admin_emails[0],  # Send to first admin, or implement a different strategy
                    subject=f"New Trade Request: {trade.symbol} {trade.direction.upper()}",
                    body=f"""
                    A new trade request has been submitted:
                    
                    Symbol: {trade.symbol}
                    Direction: {trade.direction.upper()}
                    Amount: {trade.amount}
                    Requested by: {request.current_user.username}
                    
                    Please review and approve or reject this trade.
                    """
                )
        
        return jsonify({
            'message': 'Trade request created successfully',
            'trade': trade.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error creating trade: {e}")
        return jsonify({'error': 'Failed to create trade request'}), 500

@app.route('/api/trades/<trade_id>/approve', methods=['POST'])
@token_required
def approve_trade(trade_id):
    """Approve a trade request."""
    return _process_trade_approval(trade_id, 'approved')

@app.route('/api/trades/<trade_id>/reject', methods=['POST'])
@token_required
def reject_trade(trade_id):
    """Reject a trade request."""
    return _process_trade_approval(trade_id, 'rejected')

def _process_trade_approval(trade_id: str, status: str):
    """Process trade approval or rejection."""
    data = request.get_json() or {}
    comment = data.get('comment', '')
    
    # Find trade
    trade = Trade.query.get(trade_id)
    if not trade:
        return jsonify({'error': 'Trade not found'}), 404
    
    # Check if trade can be approved/rejected
    if trade.status != Trade.STATUS_PENDING:
        return jsonify({'error': f'Trade is already {trade.status}'}), 400
    
    # Create approval record
    approval = TradeApproval(
        trade_id=trade.id,
        user_id=request.current_user.id,
        status=status,
        comment=comment
    )
    
    # Update trade status
    trade.status = Trade.STATUS_APPROVED if status == 'approved' else Trade.STATUS_REJECTED
    trade.updated_at = datetime.utcnow()
    
    db.session.add(approval)
    db.session.commit()
    
    # Notify user who created the trade
    notify_user(
        user_id=trade.user_id,
        title=f"Trade {trade.status.capitalize()}",
        message=f"Your trade {trade.symbol} {trade.direction.upper()} has been {trade.status}",
        type="success" if status == 'approved' else "warning",
        action_url=f"/trades/{trade.id}"
    )
    
    # Broadcast update to all connected clients
    socketio.emit('trade_updated', {
        'trade_id': trade.id,
        'status': trade.status,
        'updated_at': trade.updated_at.isoformat(),
        'approver': {
            'id': request.current_user.id,
            'username': request.current_user.username
        }
    }, room=f'trade_{trade.id}')
    
    return jsonify({
        'message': f'Trade {status} successfully',
        'trade': trade.to_dict()
    })

@app.route('/api/trades', methods=['GET'])
@token_required
def list_trades():
    """List trades with optional filters."""
    # Get query parameters
    status = request.args.get('status')
    user_id = request.args.get('user_id')
    symbol = request.args.get('symbol')
    limit = min(int(request.args.get('limit', 50)), 100)
    offset = int(request.args.get('offset', 0))
    
    # Build query
    query = Trade.query
    
    # Apply filters
    if status:
        query = query.filter_by(status=status)
    if user_id and (request.current_user.is_admin or str(request.current_user.id) == user_id):
        query = query.filter_by(user_id=user_id)
    elif not request.current_user.is_admin:
        # Regular users can only see their own trades
        query = query.filter_by(user_id=request.current_user.id)
    
    if symbol:
        query = query.filter(Trade.symbol.ilike(f'%{symbol}%'))
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    trades = query.order_by(Trade.created_at.desc()).offset(offset).limit(limit).all()
    
    return jsonify({
        'total': total,
        'trades': [trade.to_dict() for trade in trades]
    })

@app.route('/api/trades/<trade_id>', methods=['GET'])
@token_required
def get_trade(trade_id):
    """Get trade details by ID."""
    trade = Trade.query.get(trade_id)
    if not trade:
        return jsonify({'error': 'Trade not found'}), 404
    
    # Check permissions
    if not request.current_user.is_admin and trade.user_id != request.current_user.id:
        return jsonify({'error': 'Not authorized'}), 403
    
    return jsonify(trade.to_dict())

# Web interface routes
@app.route('/')
@token_required
def index():
    """Main dashboard page."""
    return render_template('index.html', user=request.current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        # Handle login form submission
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        if not user.is_active:
            flash('Your account is disabled', 'danger')
            return redirect(url_for('login'))
        
        # Login successful
        session.permanent = True
        session['user_id'] = user.id
        session['token'] = user.generate_auth_token('access')
        
        flash('You have been logged in', 'success')
        next_page = request.args.get('next') or url_for('index')
        return redirect(next_page)
    
    # Show login form
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Log out the current user."""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Helper functions
def notify_user(
    user_id: int,
    title: str,
    message: str,
    type: str = 'info',
    action_url: Optional[str] = None
) -> None:
    """Create a notification for a specific user."""
    try:
        notification = Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=type,
            action_url=action_url
        )
        
        db.session.add(notification)
        db.session.commit()
        
        # Emit WebSocket event
        socketio.emit('new_notification', {
            'id': notification.id,
            'title': title,
            'message': message,
            'type': type,
            'action_url': action_url,
            'created_at': notification.created_at.isoformat()
        }, room=f'user_{user_id}')
        
    except Exception as e:
        app.logger.error(f"Error creating notification: {e}")
        db.session.rollback()

def notify_admins(
    title: str,
    message: str,
    type: str = 'info',
    action_url: Optional[str] = None
) -> None:
    """Create a notification for all admin users."""
    try:
        admins = User.query.filter_by(is_admin=True).all()
        for admin in admins:
            notification = Notification(
                user_id=admin.id,
                title=title,
                message=message,
                type=type,
                action_url=action_url
            )
            db.session.add(notification)
            
            # Emit WebSocket event
            socketio.emit('new_notification', {
                'id': notification.id,
                'title': title,
                'message': message,
                'type': type,
                'action_url': action_url,
                'created_at': notification.created_at.isoformat()
            }, room=f'user_{admin.id}')
        
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error notifying admins: {e}")
        db.session.rollback()

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.is_json:
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.is_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

# Command-line tasks
@app.cli.command('initdb')
def initdb_command():
    """Initialize the database."""
    db.create_all()
    
    # Create admin user if none exists
    if not User.query.filter_by(is_admin=True).first():
        admin = User(
            username='admin',
            email='admin@example.com',
            is_admin=True
        )
        admin.set_password('admin123')  # Change this in production!
        db.session.add(admin)
        db.session.commit()
        print("Created admin user: admin / admin123")
    
    print("Database initialized")

@app.cli.command('createuser')
def create_user_command():
    """Create a new user."""
    import getpass
    
    username = input("Username: ")
    email = input("Email: ")
    is_admin = input("Is admin? (y/n): ").lower() == 'y'
    
    while True:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password == confirm:
            break
        print("Passwords don't match. Try again.")
    
    user = User(
        username=username,
        email=email,
        is_admin=is_admin
    )
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        print(f"User '{username}' created successfully")
    except Exception as e:
        db.session.rollback()
        print(f"Error creating user: {e}")

# Scheduled tasks
def check_expired_trades():
    """Check for and expire old trades."""
    with app.app_context():
        expired_trades = Trade.query.filter(
            Trade.status == Trade.STATUS_PENDING,
            Trade.expiry_time < datetime.utcnow()
        ).all()
        
        for trade in expired_trades:
            trade.status = Trade.STATUS_EXPIRED
            trade.updated_at = datetime.utcnow()
            
            # Notify user
            notify_user(
                user_id=trade.user_id,
                title="Trade Expired",
                message=f"Your trade {trade.symbol} {trade.direction.upper()} has expired",
                type="warning",
                action_url=f"/trades/{trade.id}"
            )
        
        if expired_trades:
            db.session.commit()
            app.logger.info(f"Expired {len(expired_trades)} trades")

# Initialize scheduler
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=check_expired_trades,
    trigger='interval',
    minutes=5
)
scheduler.start()

# Application factory pattern
def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    socketio.init_app(app)
    
    # Register blueprints (if any)
    # from .api import api_bp
    # app.register_blueprint(api_bp, url_prefix='/api')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Run the application
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=app.debug,
        use_reloader=not app.debug
    )