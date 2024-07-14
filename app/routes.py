from flask import Blueprint, render_template, request, jsonify

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/swipe', methods=['POST'])
def swipe():
    data = request.get_json()
    # Process the swipe data here
    return jsonify(status='success')
