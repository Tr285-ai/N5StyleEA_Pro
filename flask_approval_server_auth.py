# flask_approval_server_auth.py
from flask import Flask, request, jsonify
from functools import wraps
import jwt
import datetime
from typing import Dict, Any, Optional
import logging
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# In-memory user database (replace with a real database in production)
users = {
    'trader1': {
        'username': 'trader1',
        'password': generate_password_hash('password123'),
        'role': 'trader'
    },
    'admin': {
        'username': 'admin',
        'password': generate_password_hash('admin123'),
        'role': 'admin'
    }
}

# In-memory trade approvals
trade_approvals = {}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in the header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
            
        try:
            # Decode the token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = users.get(data['username'])
            
            if current_user is None:
                return jsonify({'message': 'User not found'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
            
        return f(current_user, *args, **kwargs)
        
    return decorated

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint."""
    auth = request.authorization
    
    if not auth or not auth.username or not auth.password:
        return jsonify({'message': 'Could not verify'}), 401
        
    user = users.get(auth.username)
    
    if not user or not check_password_hash(user['password'], auth.password):
        return jsonify({'message': 'Invalid credentials'}), 401
        
    # Generate token
    token = jwt.encode({
        'username': user['username'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, app.config['SECRET_KEY'])
    
    return jsonify({
        'token': token,
        'user': {
            'username': user['username'],
            'role': user['role']
        }
    })

@app.route('/api/trade/request', methods=['POST'])
@token_required
def request_trade_approval(current_user):
    """Request approval for a trade."""
    data = request.get_json()
    
    required_fields = ['symbol', 'side', 'quantity', 'price']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # Generate a unique trade ID
    trade_id = str(len(trade_approvals) + 1)
    
    # Store the trade request
    trade_approvals[trade_id] = {
        'id': trade_id,
        'user': current_user['username'],
        'symbol': data['symbol'],
        'side': data['side'],
        'quantity': data['quantity'],
        'price': data.get('price'),
        'status': 'pending',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'comments': []
    }
    
    logger.info(f"Trade {trade_id} requested by {current_user['username']}")
    
    return jsonify({
        'message': 'Trade approval requested',
        'trade_id': trade_id
    }), 201

@app.route('/api/trade/approve/<trade_id>', methods=['POST'])
@token_required
def approve_trade(current_user, trade_id):
    """Approve or reject a trade (admin only)."""
    if current_user['role'] != 'admin':
        return jsonify({'message': 'Admin access required'}), 403
        
    data = request.get_json()
    action = data.get('action', '').lower()
    
    if action not in ['approve', 'reject']:
        return jsonify({'message': 'Invalid action'}), 400
        
    if trade_id not in trade_approvals:
        return jsonify({'message': 'Trade not found'}), 404
        
    trade = trade_approvals[trade_id]
    
    if trade['status'] != 'pending':
        return jsonify({'message': f'Trade is already {trade["status"]}'}), 400
        
    # Update trade status
    trade['status'] = 'approved' if action == 'approve' else 'rejected'
    trade['reviewed_by'] = current_user['username']
    trade['reviewed_at'] = datetime.datetime.utcnow().isoformat()
    trade['comment'] = data.get('comment', '')
    
    logger.info(f"Trade {trade_id} {trade['status']} by {current_user['username']}")
    
    return jsonify({
        'message': f'Trade {action}d successfully',
        'trade': trade
    })

@app.route('/api/trades', methods=['GET'])
@token_required
def get_trades(current_user):
    """Get list of trades (filtered by user unless admin)."""
    if current_user['role'] == 'admin':
        return jsonify(list(trade_approvals.values()))
    else:
        return jsonify([
            trade for trade in trade_approvals.values() 
            if trade['user'] == current_user['username']
        ])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)