from google import genai

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

