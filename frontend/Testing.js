import fs from "fs";

async function uploadFile() {

    const image = fs.readFileSync("./dummy.jpeg");
    const blob = new Blob([image], { type: "image/jpeg" })

    const formData = new FormData();
    formData.append("file", blob, "dummy.jpeg")

    const result = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData
    });

    const data = await result.json();

    console.log(data)
}

uploadFile();