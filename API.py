from flask import Flask, request, jsonify, send_from_directory
from chatbot import generate_langchain_response, sales_data # Import functions from chatbot.py

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('', 'index.html')

@app.route('/api/rep_performance', methods=['GET'])
def rep_performance():
    rep_id = int(request.args.get('rep_id'))
    rep_data = sales_data[sales_data['employee_id'] == rep_id].iloc[0]
    response = generate_langchain_response(f"Performance of rep {rep_id}")
    return jsonify({"performance_feedback": response})

@app.route('/api/team_performance', methods=['GET'])
def team_performance():
    response = generate_langchain_response("Overall team performance")
    return jsonify({"team_performance_feedback": response})

@app.route('/api/performance_trends', methods=['GET'])
def performance_trends():
    time_period = request.args.get('time_period')
    response = generate_langchain_response(f"Performance trends for {time_period}")
    return jsonify({"performance_trends_feedback": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)
