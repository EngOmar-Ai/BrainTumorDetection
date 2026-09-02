import fs from "fs";

async function uploadFile(){

    const image = fs.readFileSync("./dummy.jpeg");
    const blob = new Blob([image], { type: "image/jpeg" })

    const formData = new FormData();
    formData.append("file", blob, "dummy.jpeg")

    const result = await fetch("http://localhost:8000/sessions/initiate", {
        method: "POST",
        body: formData
    });

    const data = await result.json();

    console.log(data.id);
    console.log(data.response)

    return data
}

async function sendMessage(sessionId, message) {
    const response = await fetch('http://localhost:8000/sessions/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            message: message
        })
    });

    const data = await response.json();

    console.log(data.response);

    return data.response; // Returns the assistant's reply string
}

const data = uploadFile();
const ai = sendMessage(" -- The ID Goes Here --","But What is Meningioma, That Sounds Scary, Should I Be Worried?")
