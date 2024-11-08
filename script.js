function highlightPDF(pdfPath, pageNumber, textToHighlight) {
    // Open a new window to display the PDF
    var pdfWindow = window.open("", "_blank", "width=800,height=600");
    pdfWindow.document.write(`
        <iframe src="${pdfPath}#page=${pageNumber}" width="100%" height="100%" id="pdfViewer"></iframe>
        <script>
            // Wait for the PDF to load and then highlight the specific text
            window.onload = function() {
                setTimeout(function() {
                    let iframe = document.getElementById("pdfViewer");
                    let innerDoc = iframe.contentDocument || iframe.contentWindow.document;
                    
                    // Find and highlight the text in the PDF (this will be a simplified implementation)
                    let spans = innerDoc.querySelectorAll("span");
                    spans.forEach(function(span) {
                        if(span.textContent.includes("${textToHighlight}")) {
                            span.style.backgroundColor = "red";
                            span.style.color = "white";
                        }
                    });
                }, 1000); // Delay for PDF to render before highlighting
            };
        </script>
    `);
}