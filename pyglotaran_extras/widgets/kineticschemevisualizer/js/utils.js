function exportGraphAsPng(cy, options = {}) {
  const defaultOptions = {
    output: 'base64uri',
    bg: 'white'
  };
  
  const exportOptions = Object.assign({}, defaultOptions, options);
  return cy.png(exportOptions);
}

function createExportButton(cy) {
  let button = document.createElement("button");
  button.textContent = "Export PNG";
  button.addEventListener("click", () => {
      let pngData = exportGraphAsPng(cy);
      let downloadLink = document.createElement("a");
      downloadLink.href = pngData;
      downloadLink.download = "graph.png";
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
  });
  return button;
}

function createRadio(id, name, value, className, applyLayout, cy) {
  let radio = document.createElement("input");
  radio.type = "radio";
  radio.id = id;
  radio.name = name;
  radio.value = value;
  radio.classList.add(className);

  radio.addEventListener("change", () => {
      if (radio.checked) {
          applyLayout(cy);
      }
  });

  return radio;
}

function createLabel(id, textContent, className) {
  let label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = textContent;
  label.classList.add(className);

  return label;
}

module.exports = {
  exportGraphAsPng,
  createExportButton,
  createRadio,
  createLabel,
}
