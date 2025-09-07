import os
import json
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph , START , END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class ReviewsState(TypedDict):
    email:str
    Review : str
    SentimentReason:str
    Sentiment:str
    if_pos_responce : str
    if_negative_department:str
    if_negative_then_responce_to_user:str
    if_negative_then_responce_to_department:str

class sentimentstruct(BaseModel):
    SentimentReason:str = Field(..., description="A reason for the sentiment in 20 words.")
    Sentiment:Literal["Positive" , "Negative" , "Neutral"] = Field(...)
    Email:str = Field(...)

class Negative_review(BaseModel):
    department:Literal["Ticketing" , "Schedule" , "Food and Beverage"] = Field(...)
    reason : str = Field(..., description="The reason for the issue in 50 words.")
    suggestions: str = Field(...)

gemini_api_key = os.getenv("GEMINI")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash" , api_key=gemini_api_key)
LlmStructureOutput_sentiment = llm.with_structured_output(sentimentstruct)
LlmStructureOutput_Negative_review_depart = llm.with_structured_output(Negative_review)

Review  = "my email is kingfahad@gmail.com After having lived in Africa for more than seven years (Mozambique) in addition to now frequently traveling for work I thought I had seen everything. In terms of poor management, traveling with Air Senegal was surprisingly bad. No communication or information, clueless staff and no compensation or willingness to assist passengers (elderly, young, families etc.). My flight from Dakar to Praia was 5hrs late, which was more than the average delay on their flights (3.5hrs). The aircraft supposed to take us to Praia departed for Conakry at the same time as we were supposed to take-off. After speaking with ground staff it became clear that it’s a management strategy/policy - they simply plan more flights (thus, sell more tickets) than their fleet is capable of handling. The operation side of the company have their hands tied as they dont have enough aircrafts or staff to operate according to the route network and timetable, which is reflected on their constant delays."

def check_output_type(state:ReviewsState)->ReviewsState:
    review = state["Review"]
    StructOutputResponce = LlmStructureOutput_sentiment.invoke(review)
    return {"SentimentReason":StructOutputResponce.SentimentReason , "Sentiment":StructOutputResponce.Sentiment , "email" : StructOutputResponce.Email}

def positiveResponce(state : ReviewsState)->ReviewsState:
    review = state["Review"]
    prompt = f"you are required to answer the flight costomer reviw which is positive so you will write responce to user in thakfull polite {review} incase any sugestion you align your respoce that we will not disturbs you again "
    responce = llm.invoke(prompt)
    return{"if_pos_responce":responce.content}

def NegativeResponce(state: ReviewsState) -> ReviewsState:
    review = state["Review"]
    customer_prompt = (
        f"Write a polite, fruitful, and professional response to tackle the situation ,The customer gave a negative review: {review}. psychologically and beautifully, so the customer remains part of our airline. thank them for being with us, and show we value their feedback.Their email is {state['email']}"
    )
    user_response = llm.invoke(customer_prompt).content
    department_info = LlmStructureOutput_Negative_review_depart.invoke(review)
    
    return {
        "if_negative_department": department_info.department,
        "if_negative_then_responce_to_user": user_response,
        "if_negative_then_responce_to_department": f"Reason: {department_info.reason}\nSuggestions: {department_info.suggestions}"
    }

def RequestToCorespondingDepartment(
    state: ReviewsState
) -> Literal["ticketing_department", "Scheduling_department", "food_department"]:
    dept = state.get("if_negative_department", "").lower()
    if "ticket" in dept:
        return "ticketing_department"
    elif "schedule" in dept:
        return "Scheduling_department"
    elif "food" in dept:
        return "food_department"
    else:
        return "ticketing_department"

def food_department(state: ReviewsState) -> ReviewsState:
    email = state["email"]
    reason = state["if_negative_then_responce_to_department"]
    dept_report = (
        f"Internal Report: Customer with email {email} had an issue "
        f"in Food department. Reason: {reason}. "
        "Action: Please investigate and suggest improvements."
    )
    user_response = llm.invoke(dept_report)
    return {}

def Scheduling_department(state: ReviewsState) -> ReviewsState:
    email = state["email"]
    reason = state["if_negative_then_responce_to_department"]
    dept_report = (
        f"Internal Report: Customer with email {email} had an issue "
        f"in Scheduling department. Reason: {reason}. "
        "Action: Please investigate and suggest improvements."
    )
    user = llm.invoke(dept_report)
    return {}

def ticketing_department(state: ReviewsState) -> ReviewsState:
    email = state["email"]
    reason = state["if_negative_then_responce_to_department"]
    dept_report = f"Internal Report: Customer with email {email} had an issue in Ticketing department. Reason: {reason}. Action: Please investigate and suggest improvements."
    user = llm.invoke(dept_report)
    return {}

def CheckResponce_pos_neg(
    state: ReviewsState
) -> Literal["positiveResponce", "NegativeResponce"]:
    if state["Sentiment"].lower() == "positive":
        return "positiveResponce"
    else:
        return "NegativeResponce"

graph = StateGraph(ReviewsState)

graph.add_node("check_output_type", check_output_type)
graph.add_node("positiveResponce", positiveResponce)
graph.add_node("NegativeResponce", NegativeResponce)
graph.add_node("ticketing_department", ticketing_department)
graph.add_node("food_department", food_department)
graph.add_node("Scheduling_department", Scheduling_department)

graph.add_edge(START, "check_output_type")
graph.add_conditional_edges("check_output_type", CheckResponce_pos_neg, {
    "positiveResponce": "positiveResponce",
    "NegativeResponce": "NegativeResponce"
})

graph.add_edge("positiveResponce", END) 

graph.add_conditional_edges("NegativeResponce", RequestToCorespondingDepartment, {
    "ticketing_department": "ticketing_department",
    "Scheduling_department": "Scheduling_department",
    "food_department": "food_department"
})

graph.add_edge("ticketing_department", END)
graph.add_edge("food_department", END)
graph.add_edge("Scheduling_department", END)

workflow = graph.compile()

initial_state_org = {
    "email" :"",
    "Review" : Review,
    "SentimentReason":"",
    "Sentiment":"",
    "if_pos_responce" : "",
    "if_negative_department":"",
    "if_negative_then_responce_to_user":"",
    "if_negative_then_responce_to_department":""
}

final_state_org = workflow.invoke(initial_state_org)
print(final_state_org)