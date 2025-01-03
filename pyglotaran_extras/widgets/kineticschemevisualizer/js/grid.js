const cytoscape = require('cytoscape');
const gridGuide = require('cytoscape-grid-guide');

gridGuide( cytoscape );

function applyGridGuide(cytoscapeInstance) {
    var options = {
        snapToGridOnRelease: false,
        snapToGridDuringDrag: true,
        snapToAlignmentLocationOnRelease: false,
        snapToAlignmentLocationDuringDrag: false,
        distributionGuidelines: false,
        geometricGuideline: false,
        initPosAlignment: false,
        centerToEdgeAlignment: false,
        resize: false,
        parentPadding: false,
        drawGrid: true,
        gridSpacing: 10,
        snapToGridCenter: true,
        zoomDash: true,
        panGrid: false,
        gridStackOrder: -1,
        gridColor: '#dedede',
        lineWidth: 1.0,
        guidelinesStackOrder: 4,
        guidelinesTolerance: 2.00,
        guidelinesStyle: {
            strokeStyle: "#8b7d6b",
            geometricGuidelineRange: 400,
            range: 100,
            minDistRange: 10,
            distGuidelineOffset: 10,
            horizontalDistColor: "#ff0000",
            verticalDistColor: "#00ff00",
            initPosAlignmentColor: "#0000ff",
            lineDash: [0, 0],
            horizontalDistLine: [0, 0],
            verticalDistLine: [0, 0],
            initPosAlignmentLine: [0, 0],
        },
        parentSpacing: -1
    };

    cytoscapeInstance.gridGuide(options);
}

module.exports = {
    applyGridGuide
}