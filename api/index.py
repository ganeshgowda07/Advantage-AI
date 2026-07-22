import os
import json
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from groq import Groq

# Load local environment variables if present
load_dotenv()

# Point to the root-level templates directory relative to the serverless api function
app = Flask(__name__, template_folder='../templates')

# Initialize API Keys securely from the production environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

@app.route('/')
def home():
    """Renders the main single-page sports coach dashboard."""
    return render_template(
        'index.html', 
        supabase_url=SUPABASE_URL, 
        supabase_anon_key=SUPABASE_ANON_KEY
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Handshake validation endpoint for the frontend client."""
    return jsonify({
        "status": "healthy",
        "environment": "US-Sports-Ecosystem",
        "framework": "Flask Serverless",
        "api_configured": bool(GROQ_API_KEY)
    })

@app.route('/api/debrief', methods=['POST'])
def match_debrief():
    """Processes match narratives and returns structured tactical insights via Groq."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        narrative = data.get("narrative", "").strip()
        error_type = data.get("error_type", "Unforced errors")
        opponent_type = data.get("opponent_type", "Baseliner")
        player_level = data.get("player_level", "High School Varsity / NTRP 3.5")

        if not narrative:
            return jsonify({"error": "Match narrative input payload cannot be blank."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an expert US Tennis Solutions Architect and Strategic Performance Coach specializing "
            "in the US High School (Varsity/JV) and NCAA competitive ecosystem. Analyze the player's match performance "
            "narrative, main error classification, and opponent playstyle based on standard NTRP guidelines. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown blocks outside it:\n"
            "{\n"
            "  \"readiness_score\": 78,\n"
            "  \"motivational_tip\": \"Keep your feet active through the baseline transitions. Next set is yours!\",\n"
            "  \"weaknesses\": [\"Explanation of tactical breakdown 1\", \"Explanation of tactical breakdown 2\"],\n"
            "  \"tactics\": [\"Actionable adjustment 1\", \"Actionable adjustment 2\", \"Actionable adjustment 3\"]\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Opponent Archetype: {opponent_type}\n"
            f"Primary Error Focus: {error_type}\n"
            f"Match Narrative: {narrative}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Analysis pipeline failure: {str(e)}"}), 500

@app.route('/api/training', methods=['POST'])
def generate_training():
    """Generates a structured US tennis drill, fitness schedule, and standard high-performance nutritional plan."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        focus_area = data.get("focus_area", "Serves")
        time_available = data.get("time_available", "30 mins a day")
        gym_access = data.get("gym_access", "Full gym access")
        player_level = data.get("player_level", "High School Varsity / NTRP 3.5")

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an elite US tennis performance coordinator and high-performance sports nutritionist. "
            "Create a target development schedule using standard US metrics, common athletic fitness terms, and widely available US nutritional staples. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown formatting:\n"
            "{\n"
            "  \"drills\": [\"Drill 1 Description\", \"Drill 2 Description\"],\n"
            "  \"gym_exercises\": [\"Exercise 1 Description\", \"Exercise 2 Description\"],\n"
            "  \"pre_match_meals\": [\"Pre-Match Meal Combo 1\", \"Pre-Match Meal Combo 2\"],\n"
            "  \"post_match_meals\": [\"Post-Match Recovery 1\", \"Post-Match Recovery 2\"]\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Target Technical Focus: {focus_area}\n"
            f"Daily Time Constraints: {time_available}\n"
            f"Physical Gym Infrastructure Setup: {gym_access}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Training hub pipeline failure: {str(e)}"}), 500

@app.route('/api/recovery', methods=['POST'])
def injury_recovery():
    """Generates a structured day-to-day injury recovery timeline and advice via Groq."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        injury = data.get("injury", "").strip()
        context = data.get("context", "").strip()
        player_level = data.get("player_level", "Ordinary / Recreational")

        if not injury:
            return jsonify({"error": "Injury type or classification name cannot be blank."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an elite US sports physical therapist, athletic trainer, and tennis injury rehabilitation specialist. "
            "Analyze the player's tennis-related pain or injury context and produce a structured, safe, progressive recovery schedule. "
            "Keep the suggestions clinical, practical, and highly focused on tennis bio-mechanics. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown formatting:\n"
            "{\n"
            "  \"injury_name\": \"Injury Title / Area\",\n"
            "  \"recovery_plan\": [\n"
            "    {\n"
            "      \"day\": \"Phase or Day Range (e.g., Day 1-2: Acute Protection)\",\n"
            "      \"focus\": \"Short, clinical clinical target (e.g., Swelling Control & Tissue Calm)\",\n"
            "      \"protocols\": [\"Protocol action item 1\", \"Protocol action item 2\", \"Protocol action item 3\"]\n"
            "    }\n"
            "  ],\n"
            "  \"disclaimer\": \"Informational therapeutic outline only. Always seek counsel from an athletic medical professional before returning to court.\"\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Reported Injury/Pain: {injury}\n"
            f"Symptoms & Context Details: {context}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Injury recovery pipeline failure: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def technical_chat():
    """Processes interactive user queries regarding drills, settings, and general tennis strategies."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"error": "Chat prompt value cannot be empty."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are AdvantageAI, an expert 24/7 technical tennis mentor and strategy guide built for US high school "
                    "and college competitive tennis athletes. Answer follow-up queries with direct tactical solutions, "
                    "mental framework improvements, or mechanical execution suggestions. Keep answers concise and strictly actionable."
                )
            }
        ]
        
        for turn in history:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
            
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )

        return jsonify({"response": response.choices[0].message.content.strip()})

    except Exception as e:
        return jsonify({"error": f"Interactive chat system breakdown: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)