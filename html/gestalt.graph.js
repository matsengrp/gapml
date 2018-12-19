var legend_width = 100;
var legend_height = 150;
// legend spacer between individual entries
var legendSpace = 20
var legendOffset = 10

var width = 6400;
var height = 4800;
var time_offset = 100;
var y_pos_offset = 100;
var x_offset = 100;
var y_offset = 100;
var node_radius = 30;
var pie_radius = 5;
var COLORS = [
  "#4F6128", //Brain
  "#77933C", //Eye
  "#FFC000", //Gills
  "#632523", //Heart
  "#558ED5", //Intestinal
];

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

function build_graph(data, options) {
  drawLegend(data.tissues);

  var nodes = data.nodes;
  var links = data.links;
  var svg = d3.select('body').append('svg')
    .attr('width', width)
    .attr('height', height);

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
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("stroke-width", function(d){
      return d.value;
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
          pg.radius = (Math.log(d.num_uniqs) + 1) * pie_radius;
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
    link.attr('x1', function(d) { return d.source.time_idx * time_offset + x_offset; })
        .attr('y1', function(d) { return d.source.yPos * y_pos_offset + y_offset; })
        .attr('x2', function(d) { return d.target.time_idx * time_offset + x_offset; })
        .attr('y2', function(d) { return d.target.yPos * y_pos_offset + y_offset; });

  });

  force.start();
}

// populate the dropdown with options
var allTrees = "";
$.ajax({
  url: 'test_master_list.json',
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
  allTrees.forEach(function(entry) {
    if (entry.tree_file == optionSelected){
      current_data_settings = entry;
    }
  })

  // First clean up the canvas
  d3.select("svg").remove();

  // Now load data and draw
  d3.json(current_data_settings.tree_file , function(error, treeData) {
    build_graph(treeData, current_data_settings);
  });
};
