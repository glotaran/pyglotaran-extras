const cytoscape = require("cytoscape");
var panzoom = require("cytoscape-panzoom");

panzoom(cytoscape);

const { getDagreLayoutOptions, applyGridLayout } = require("./layouts");
const { createExportButton, createLabel, createRadio } = require("./utils");
const { applyGridGuide } = require("./grid");

function createNodeStyle(visualization_options) {
  return {
    "background-color": function (ele) {
      let nodeId = ele.data("id");
      let nodeColor = ele.data("bg"); // Default color
      if (visualization_options.hasOwnProperty("colour_node_mapping")) {
        let colorMapping = visualization_options["colour_node_mapping"];
        for (const [color, ids] of Object.entries(colorMapping)) {
          if (ids.includes(nodeId)) {
            nodeColor = color;
            break;
          }
        }
      }
      return nodeColor;
    },
    color: function (ele) {
      let nodeId = ele.data("id");
      let textColor = "black"; // Default text color
      if (visualization_options.hasOwnProperty("colour_node_mapping")) {
        let colorMapping = visualization_options["colour_node_mapping"];
        for (const [color, ids] of Object.entries(colorMapping)) {
          if (
            ids.includes(nodeId) &&
            (color === "black" || color === "brown" || color === "blue")
          ) {
            textColor = "white";
            break;
          }
        }
      }
      return textColor;
    },
    label: function (ele) {
      let nodeId = ele.data("id");
      let nodesOptions = visualization_options["nodes"] || {};
      return nodesOptions[nodeId] && nodesOptions[nodeId]["alternate_name"]
        ? nodesOptions[nodeId]["alternate_name"]
        : nodeId;
    },
    shape: "rectangle",
    width: function (ele) {
      let nodeId = ele.data("id");
      let nodesOptions = visualization_options["nodes"] || {};
      return nodesOptions[nodeId] && nodesOptions[nodeId]["width"]
        ? nodesOptions[nodeId]["width"]
        : 80;
    },
    height: function (ele) {
      let nodeId = ele.data("id");
      let nodesOptions = visualization_options["nodes"] || {};
      return nodesOptions[nodeId] && nodesOptions[nodeId]["height"]
        ? nodesOptions[nodeId]["height"]
        : 30;
    },
    opacity: function (ele) {
      return ele.data("id").includes("GS") ? 0 : 1;
    },
    "text-halign": "center",
    "text-valign": "center",
  };
}

function render({ model, el }) {
  let graph_data = model.get("graph_data");
  let visualization_options = model.get("visualization_options");
  console.log(graph_data);
  console.log(visualization_options);
  let widgetHeight = model.get("height").toString();

  let container = document.createElement("div");
  container.classList.add("container");
  container.style.height = widgetHeight + "px";

  let graphContainer = document.createElement("div");
  graphContainer.classList.add("graph-container");

  let controlsContainer = document.createElement("div");
  controlsContainer.classList.add("controls-container");

  container.appendChild(graphContainer);
  container.appendChild(controlsContainer);

  el.appendChild(container);

  let cy = cytoscape({
    container: graphContainer,
    elements: graph_data.hasOwnProperty("elements")
      ? graph_data["elements"]
      : {},
    userZoomingEnabled: false,
    style: cytoscape
      .stylesheet()
      .selector("node")
      .style(createNodeStyle(visualization_options))
      .selector("edge")
      .style({
        width: 3,
        "line-color": "#ccc",
        "target-arrow-color": "#ccc",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        label: "data(weight)",
      })
      .selector(""),
    layout:
      graph_data["elements"] &&
      graph_data["elements"]["nodes"] &&
      graph_data["elements"]["nodes"][0] &&
      graph_data["elements"]["nodes"][0]["position"] &&
      "x" in graph_data["elements"]["nodes"][0]["position"] &&
      "y" in graph_data["elements"]["nodes"][0]["position"]
        ? { name: "preset" }
        : getDagreLayoutOptions(),
  });

  cy.panzoom();

  let exportButton = createExportButton(cy);
  controlsContainer.appendChild(exportButton);
  model.set("cy_json", cy.json());
  model.save_changes();

  cy.on("free ", (event) => {
    console.log("free event triggered", event);
    model.set("cy_json", cy.json());
    model.save_changes();
  });

  model.on("change:visualization_options", () => {
    console.log("Updating style");
    cy.style()
      .selector("node")
      .style(createNodeStyle(model.get("visualization_options")))
      .update();
  });

  let exportJsonButton = document.createElement("button");
  exportJsonButton.textContent = "Export JSON";
  exportJsonButton.addEventListener("click", () => {
    let jsonData = JSON.stringify(cy.json());
    let dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(jsonData);
    let downloadLink = document.createElement("a");
    downloadLink.href = dataUri;
    downloadLink.download = "graph.json";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  });
  controlsContainer.appendChild(exportJsonButton);

  applyGridGuide(cy);
}

export default { render };
