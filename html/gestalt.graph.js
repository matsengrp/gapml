var legend_width = 100;
var legend_height = 150;
// legend spacer between individual entries
var legendSpace = 20
var legendOffset = 10

// Size of svg
var width = 6400;
var height = 4800;

// Offsets between nodes
var time_offset = 100;
var y_pos_offset = 100;

// Offset from side of the window
var x_offset = 100;
var y_offset = 100;
var pie_radius_unit = 5;
var stroke_width_unit = pie_radius_unit;

var TOTAL_TISSUES = 5;
var COLORS = [
  "#4F6128", //Brain
  "#77933C", //Eye
  "#FFC000", //Gills
  "#632523", //Heart
  "#558ED5", //Intestinal
];

var ORGAN_INDICES = {
  "Brain": 0,
  "Eyes": 1,
  "Gills": 2,
  "Heart": 3,
  "Int": 4,
}

function drawLegend(sorted_tissues) {
  var colorMap = {};
  for(var i = 0; i < COLORS.length; i++){
    colorMap[sorted_tissues[i]] = COLORS[i];
  }
  // remove any previous legends
  $( "#fixeddivleg" ).empty();

  var htmlOutput = "<b>Legend: </b><br>"
  $( '#fixeddivleg' ).html( htmlOutput );

  // make a new SVG for the legend and text
  var legend = d3.select("#fixeddivleg")
    .append("svg")
    .attr("width", legend_width)
    .attr("height", legend_height)
    .append("svg:g")
    .attr("transform","translate(" + 20 + "," + 20 + ")");

  var rects = legend.selectAll("circle")
    .data(d3.entries(colorMap))
    .enter()
    .append("rect")
    .attr("x", 30)
    .attr("y", function(d, i) {
      return i * legendSpace;
    })
    .attr("fill", function(d, i) {
      return(d.value);
    })
    .attr("width", 10)
    .attr("height", 10);


  legend.selectAll("text")
    .data(d3.entries(colorMap))
    .enter()
    .append("text")
    .filter(function(d) {
      return d.key !== "UNKNOWN"
    })
    .attr("x",  30 + legendSpace)
    .attr("y", function(d, i) {
      return i * legendSpace + legendOffset;
    })
    .text(function(d,i){
      return d.key;
    });
};

function build_graph(data, svg, organSelected) {
  var nodes = data.nodes;
  var links = data.links;

  // Is the primary calculator for placement of graph
  var force = d3.layout.force()
    .size([width, height])
    .nodes(nodes)
    .links(links);

  // Associate links with data
  var link = svg.selectAll('.link')
    .data(links)
    .enter()
    .append('line')
    .attr('class', 'link')
    .attr("stroke-width", function(d){
      return (Math.log(d.value) + 1) * stroke_width_unit;
    });

  // Associate nodes with data
  var node = svg.selectAll('.node')
    .data(nodes)
    .enter()
    .append('g')
    .attr('class', 'pies');

  // Create generator for pie charts
  var pie = d3.layout.pie()
  		.sort(null)
  		.value(function(d) { return d; });

  // Create generator for pie slices
  var arc = d3.arc()
			.innerRadius(0)
      .outerRadius(function (d) {
        return d.radius;
      });

  // only using the force thing to draw links... that's it
  force.on('end', function() {
    // Move nodes to appropriately places on the plot
    node.attr("transform",function(d) {
          var xPos = d.time_idx * time_offset + x_offset;
          var yPos = d.yPos * y_pos_offset + y_offset;
          return "translate("+ xPos + "," + yPos + ")";
    });

    // Create my pie charts -- radius is proportional to log of number of unique clt nodes
    var pies = node.selectAll(".pies")
      .data(function(d) {
        pie_gen = pie(d.tissues);
        pie_gen.forEach(function(pg) {
          pg.radius = (Math.log(d.num_uniqs) + 1) * pie_radius_unit;
        })
        return pie_gen;
      })
  		.enter()
  		.append('g')
  		.attr('class','arc');

    // Create the slices in the pie
    pies.append("path")
  	  .attr('d',arc)
      .attr("fill",function(d,i){
           return COLORS[i];
      });

    // update link locations between nodes
    link.attr("stroke", function(d){
        var organ_idx = ORGAN_INDICES[organSelected];
        var containsSelected = d.source.tissues[organ_idx] > 0;
        if (containsSelected) {
          return COLORS[organ_idx];
        } else {
          return "#999";
        }
      }).attr("stroke-opacity", function(d){
        var num_tissues = d.source.name.split("_")[0].length;
        var organ_idx = ORGAN_INDICES[organSelected];
        var containsSelected = d.source.tissues[organ_idx] > 0;
        if (containsSelected) {
          return 0.95 * 1/num_tissues;
        } else {
          return 0.3;
        }
      }).attr('x1', function(d) { return d.source.time_idx * time_offset + x_offset; })
      .attr('y1', function(d) { return d.source.yPos * y_pos_offset + y_offset; })
      .attr('x2', function(d) { return d.target.time_idx * time_offset + x_offset; })
      .attr('y2', function(d) { return d.target.yPos * y_pos_offset + y_offset; });

  });

  force.start();
}

function label_graph(data, svg){
  var time_max = _.max(data.nodes, function(node){
    return node.time_idx;
  }).time_idx;
  var y_max = _.max(data.nodes, function(node){
    return node.yPos;
  }).yPos;

  var yPos_labels = {};
  data.nodes.forEach(function(node_dict){
    progenitor_type = node_dict.name.split("_")[0];
    yPos_labels[progenitor_type] = node_dict.yPos;
  });

  svg.selectAll("p.ylabel")
    .data(d3.entries(yPos_labels))
    .enter()
    .append("text")
    .attr("transform",function(d) {
      var xPos = (time_max + 1) * time_offset + x_offset;
      var yPos = d.value * y_pos_offset + y_offset;
      return "translate("+ xPos + "," + yPos + ")";
    })
    .text(function(d,i){
      return d.key;
    });

  svg.selectAll("p.xlabel")
    .data(_.range(time_max + 1))
    .enter()
    .append("text")
    .attr("transform",function(d) {
      var xPos = d * time_offset + x_offset;
      var yPos = (y_max + 1) * y_pos_offset + y_offset;
      return "translate("+ xPos + "," + yPos + ")";
    })
    .text(function(d){
      return d;
    });
}

function update_plots(data, options, organSelected) {
  var svg = d3.select('#graph')
    .append('svg')
    .attr('width', width)
    .attr('height', height);
  drawLegend(data.tissues);
  label_graph(data, svg);
  build_graph(data, svg, organSelected);
}

// populate the dropdown with options
var allTrees = "";
$.ajax({
  url: "graph_master_list.json",
  async: false,
  dataType: 'json',
  success: function (response) {
    allTrees = response;
    var $choices = $("#dataSelect");
    $choices.empty();
    $.each(response, function(index, value) {
      $('#dataSelect').append($("<option></option>")
        .attr("value",value.tree_file)
        .text(value.name));
    });
  },
  error: function(XMLHttpRequest, textStatus, errorThrown) {
    alert("Status: " + textStatus); alert("Error: " + errorThrown);
  }
});

function updateData() {
  // Figure out which setting we selected
  var optionSelected = $("#dataSelect").val();
  var organSelected = $("#organSelect").val();
  allTrees.forEach(function(entry) {
    if (entry.tree_file == optionSelected){
      current_data_settings = entry;
    }
  })

  // First clean up the canvas
  d3.select('#graph').selectAll("*").remove();

  // Now load data and draw
  d3.json(current_data_settings.tree_file , function(error, treeData) {
    update_plots(treeData, current_data_settings, organSelected);
  });
};
