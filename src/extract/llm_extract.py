import os
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import WebBaseLoader
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import json
import pandas as pd

# Set API key
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# set model and model params
model = "gpt-3.5-turbo-1106"
# model="gpt-3.5-turbo-16k"
chat = ChatOpenAI(temperature=0.0, model_name=model)

# Loading multiple web links
urls = [
    "https://www.jamieoliver.com/recipes/lamb-recipes/best-roast-leg-of-lamb/",
    "https://www.bbcgoodfood.com/user/513151/recipe/malaysian-chicken-curry",
    "https://thewoksoflife.com/chinese-green-beans-pork/",
    "https://cookieandkate.com/healthy-granola-recipe/",
    "https://rasamalaysia.com/recipe-chicken-satay/",
    "https://carlsbadcravings.com/reuben-sandwich/",
    "https://dish.co.nz/recipes/tandoori-style-barbecued-chicken/",
    "http://eckitchensg.com/2017/03/hainanese-curry-rice-scissors-cut-hainanese-pork-chop/",
]
loader = WebBaseLoader(web_path=urls)

docs = loader.load()

# extraction template
name_schema = ResponseSchema(name="recipe_name", description="name of recipe")
ingredient_schema = ResponseSchema(
    name="ingredients", description="ingredient list of recipe"
)
serving_size_schema = ResponseSchema(
    name="serving_size",
    description="Extract any value or sentence about the serving size. Only report the interger value e.g. 6",
)

calorie_content_schema = ResponseSchema(
    name="calorie_per_serving",
    description="Extract any value about the calories. Only report the interger value e.g. 300",
)

fat_schema = ResponseSchema(
    name="fat_per_serving",
    description="Extract any value or sentence about the fat quantity per serving. Convert the value to grams.",
)

carb_schema = ResponseSchema(
    name="carbs_per_serving",
    description="Extract any value or sentence about the carbs quantity per serving. Convert the value to grams.",
)

protein_schema = ResponseSchema(
    name="protein_per_serving",
    description="Extract any value or sentence about the protein quantity per serving. Convert the value to grams.",
)

response_schemas = [
    name_schema,
    ingredient_schema,
    serving_size_schema,
    calorie_content_schema,
    fat_schema,
    carb_schema,
    protein_schema,
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

format_instructions = output_parser.get_format_instructions()

# print(format_instructions)

extract_template = """\
For the following text, extract the following information:

recipe_name: name of the recipe
ingredients: ingredient list of recipe.
serving_size: Extract any value about the serving size.
calorie_per_serving: Extract any value about the calories per serving.
fat_per_serving: Extract any value about the fat quantity per serving.
carbs_per_serving: Extract any value about the carb quantity per serving.
protein_per_serving: Extract any value about the protein quantity per serving.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=extract_template)

output_list = []
for i in range(len(docs)):
    doc = docs[i]
    messages = prompt.format_messages(text=doc, format_instructions=format_instructions)
    response = chat(messages)
    parsed_response = output_parser.parse(response.content)
    output_list.append(parsed_response)

json_list = json.dumps(output_list)
df = pd.read_json(json_list)

df.to_csv("../../data/processed/recipe_output.csv", index=False)
