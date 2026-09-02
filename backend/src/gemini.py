from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

def invoke(prompt: str | list) -> str:

    try:

        response = client.models.generate_content(
            model = "gemini-3.6-flash",
            contents= prompt,
        ).text

        assert response is not None

        return response

    except Exception as error:
        raise error

