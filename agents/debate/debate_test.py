from agents.debate.debate import create_debate
from agents.debate.state import DebateState


topic = "All people should have the right to own guns."

audience_members = [
    {
        "name": "Alice",
        "interests": ["technology", "innovation"],
        "work_experience": ["software engineer"],
        "personality": ["analytical", "curious"]
    },
    {
        "name": "Bob",
        "interests": ["finance", "economics"],
        "work_experience": ["banker"],
        "personality": ["pragmatic", "cautious"]
    },
    {
        "name": "Charlie",
        "interests": ["arts", "culture"],
        "work_experience": ["artist"],
        "personality": ["creative", "open-minded"]
    },
    {
        "name": "Dana",
        "interests": ["science", "research"],
        "work_experience": ["researcher"],
        "personality": ["observant", "inquisitive"]
    },
    {
        "name": "Eric",
        "interests": ["politics", "social justice"],
        "work_experience": ["activist"],
        "personality": ["passionate", "determined"]
    },
    {
        "name": "Fiona",
        "interests": ["literature", "writing"],
        "work_experience": ["editor"],
        "personality": ["thoughtful", "articulate"]
    },
    {
        "name": "George",
        "interests": ["history", "philosophy"],
        "work_experience": ["historian"],
        "personality": ["reflective", "intellectual"]
    },
    {
        "name": "Hannah",
        "interests": ["environment", "sustainability"],
        "work_experience": ["environmental scientist"],
        "personality": ["empathetic", "practical"]
    },
    {
        "name": "Ian",
        "interests": ["sports", "fitness"],
        "work_experience": ["personal trainer"],
        "personality": ["energetic", "disciplined"]
    },
    {
        "name": "Jessica",
        "interests": ["music", "performing arts"],
        "work_experience": ["musician"],
        "personality": ["expressive", "passionate"]
    },
    {
        "name": "Kevin",
        "interests": ["gaming", "technology"],
        "work_experience": ["game developer"],
        "personality": ["innovative", "strategic"]
    },
    {
        "name": "Laura",
        "interests": ["travel", "culture"],
        "work_experience": ["travel blogger"],
        "personality": ["adventurous", "curious"]
    },
    {
        "name": "Michael",
        "interests": ["politics", "debate"],
        "work_experience": ["political analyst"],
        "personality": ["assertive", "analytical"]
    },
    {
        "name": "Natalie",
        "interests": ["fashion", "design"],
        "work_experience": ["fashion designer"],
        "personality": ["creative", "stylish"]
    },
    {
        "name": "Oliver",
        "interests": ["engineering", "innovation"],
        "work_experience": ["mechanical engineer"],
        "personality": ["logical", "practical"]
    },
    {
        "name": "Patricia",
        "interests": ["health", "nutrition"],
        "work_experience": ["dietitian"],
        "personality": ["compassionate", "methodical"]
    },
    {
        "name": "Quentin",
        "interests": ["cinema", "film"],
        "work_experience": ["film critic"],
        "personality": ["observant", "expressive"]
    },
    {
        "name": "Rachel",
        "interests": ["psychology", "human behavior"],
        "work_experience": ["therapist"],
        "personality": ["empathetic", "insightful"]
    },
    {
        "name": "Samuel",
        "interests": ["technology", "cybersecurity"],
        "work_experience": ["cybersecurity expert"],
        "personality": ["cautious", "detail-oriented"]
    },
    {
        "name": "Teresa",
        "interests": ["education", "learning"],
        "work_experience": ["teacher"],
        "personality": ["patient", "nurturing"]
    },
    {
        "name": "Ulysses",
        "interests": ["literature", "poetry"],
        "work_experience": ["writer"],
        "personality": ["imaginative", "reflective"]
    },
    {
        "name": "Victoria",
        "interests": ["politics", "activism"],
        "work_experience": ["community organizer"],
        "personality": ["driven", "empathetic"]
    },
    {
        "name": "William",
        "interests": ["business", "entrepreneurship"],
        "work_experience": ["startup founder"],
        "personality": ["ambitious", "innovative"]
    },
    {
        "name": "Xander",
        "interests": ["science", "technology"],
        "work_experience": ["data scientist"],
        "personality": ["analytical", "curious"]
    },
    {
        "name": "Yvonne",
        "interests": ["art", "design"],
        "work_experience": ["graphic designer"],
        "personality": ["creative", "detail-oriented"]
    },
    {
        "name": "Zachary",
        "interests": ["sports", "business"],
        "work_experience": ["sports manager"],
        "personality": ["competitive", "driven"]
    },
    {
        "name": "Amara",
        "interests": ["social media", "marketing"],
        "work_experience": ["digital marketer"],
        "personality": ["innovative", "charismatic"]
    },
    {
        "name": "Benedict",
        "interests": ["cooking", "culinary arts"],
        "work_experience": ["chef"],
        "personality": ["creative", "passionate"]
    },
    {
        "name": "Cassandra",
        "interests": ["yoga", "meditation"],
        "work_experience": ["yoga instructor"],
        "personality": ["calm", "mindful"]
    },
    {
        "name": "Dimitri",
        "interests": ["travel", "adventure"],
        "work_experience": ["travel agent"],
        "personality": ["sociable", "adventurous"]
    }
]

pro_team_members = [
    {
        "name": "Alice",
        "expertise": "Artificial Intelligence",
        "description": "An AI specialist with deep expertise in machine learning, neural networks, and natural language processing. Advocates for strong oversight to ensure AI is developed responsibly."
    },
    {
        "name": "Bob",
        "expertise": "Ethics & Policy",
        "description": "A seasoned policy advisor with extensive experience in technology ethics, arguing that strict regulations are necessary to safeguard societal well-being."
    },
    {
        "name": "Charlie",
        "expertise": "Public Safety",
        "description": "An expert in cybersecurity and AI safety, emphasizing the need for clear regulations to prevent potential harms from unchecked AI applications."
    }
]

opp_team_members = [
    {
        "name": "Diana",
        "expertise": "Innovation & Technology",
        "description": "A technology entrepreneur who believes that heavy regulation stifles innovation and that market-driven solutions yield the best outcomes."
    },
    {
        "name": "Edward",
        "expertise": "Economics",
        "description": "An economist specializing in tech markets who argues that overregulation could hurt economic growth and limit global competitiveness."
    },
    {
        "name": "Fiona",
        "expertise": "Software Development",
        "description": "A lead software engineer and advocate for open-source innovation, stressing that flexible frameworks rather than strict rules drive technological progress."
    }
]

debate_state: DebateState = {
    "topic": topic,
    "initial_scores": [],
    "final_scores": [],
    "transcript": [],
    "round": 0,
    "audience_members": audience_members,
    "proposing_members": pro_team_members,
    "opposing_members": opp_team_members
}

debate_workflow = create_debate()
debate = debate_workflow.compile()

events = debate.stream(debate_state, stream_mode="values")

for event in events:
    print(event)
