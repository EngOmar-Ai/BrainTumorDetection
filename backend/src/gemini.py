from google.genai.errors import APIError, ServerError
from google import genai

from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "Missing Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) "
        "in your environment before starting the server."
    )

class GeminiInvocationError(Exception):
    """Raised when the Gemini call fails or returns something unusable.

    Callers should catch this specific exception (not bare Exception) and
    turn it into an appropriate HTTP response.
    """

client = genai.Client()

def invoke(prompt: str | list) -> str:

    try:

        response = client.models.generate_content(
            model = "gemini-3.6-flash",
            contents= prompt,
        ).text

        if not response:
            raise GeminiInvocationError("The Model Returned An Invalid String")

        return response

    except ServerError as error:
        raise GeminiInvocationError("Encountered A An Exception From The Gemini Servers Side") from error

    except APIError as error:
        raise GeminiInvocationError("Encountered A An Exception From The Client Side") from error

    except Exception as exception:
        raise GeminiInvocationError("Encountered A An Unexpected Exception") from exception

if __name__ == "__main__":
    ...