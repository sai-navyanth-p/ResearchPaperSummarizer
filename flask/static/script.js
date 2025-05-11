function uploadPDF() {
  const fileInput = document.getElementById("pdfFile");
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("pdf", file);

  fetch("/upload", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("summaryText").innerText = data.summary || data.error;
    })
    .catch(err => {
      console.error(err);
      alert("Failed to summarize PDF.");
    });
}
