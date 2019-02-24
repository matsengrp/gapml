/**
* the library for phylogenetic drawings
*/

/**
* JSON in master_list needs to specify the following:
* startRegion: the offsets in the aligned reads to consider for event plotting
* endRegion: the offsets in the aligned reads to consider for event plotting
*/



// ************************************************************************************************************************
//
// The main entry point for the script -- d3 tree drawing
//
// ************************************************************************************************************************

// the next three function allow for a variety of different tree branch structures
function diagonal(d, i) {
    return "M" + d.source.y + "," + d.source.x
      + "C" + (d.source.y + d.target.y) / 2 + "," + d.source.x
      + " " + (d.source.y + d.target.y) / 2 + "," + d.target.x
      + " " + d.target.y + "," + d.target.x;
}

function elbow(d, i) {
  midpointX = (d.source.x + d.target.x) / 2.0
  midpointY = (d.source.y + d.target.y) / 2.0
  return "M" + d.source.y + "," + d.source.x
  + "H" + midpointY + "V" + d.target.x + "H" + d.target.y;
}

function caterpillar(d, i) {
  midpointX = (d.source.x + d.target.x) / 2.0
  midpointY = (d.source.y + d.target.y) / 2.0
  return "M" + d.source.y + "," + d.source.x
  + "V" + d.target.x + "H" + d.target.y;
}

// recursively scale branch length
function scaleBranchLengths(nodes, w) {
  // Visit all nodes and adjust y pos width distance metric
  var visitPreOrder = function (root, callback) {
    callback(root)
    if (root.children) {
      for (var i = root.children.length - 1; i >= 0; i--) {
        visitPreOrder(root.children[i], callback)
      }
      ;
    }
  }

  visitPreOrder(nodes[0], function (node) {
    node.rootDist = node.rootDist
  })

  var rootDists = nodes.map(function (n) {
    return n.rootDist;
  });
  var yscale = d3.scale.linear()
  .domain([0, d3.max(rootDists)])
  .range([0, w]);
  visitPreOrder(nodes[0], function (node) {
    node.y = yscale(node.rootDist)
  })
  return yscale
}


// ----------------------------------------------------------------------------------------------------
// where the sausage is made -- function in functions for creating tree structures
// ----------------------------------------------------------------------------------------------------
function build_tree(selector, root, options) {  // , taxaToObj, maxCount) {
  options = options || {}

  var w = options.width - options.offset;
  h = options.height - 200,
  w = parseInt(w),
  h = parseInt(h);

  var tree = d3.layout.cluster()
  .size([h, w])
  .sort(function (node) {
    return node.children ? node.children.length : -1;
  })
  .children(function (node) {
    return node.children
  })
  .separation(function (a, b) {return 2})

  // get the legend colors
  var colorMap = legendColor(root)
  drawLegend(colorMap)

  var nodes = tree.nodes(root),
  links = tree.links(nodes);

  // spacing and locations for various graphic members
  var barHeight = options.barheight
  var event_location = options.width - (options.offset - options.barSpacer);

  var circleSize = 1.5 // radius
  var circleFill = "gray"
  var circleOutline = "black"
  var circleHighlight = "red"

  var membership_location = event_location + options.barWidth + options.barSpacer
  var barplot_location = membership_location + options.membershipBarWidth + options.barSpacer
  var counts_bar_max_size = options.countsBarWidth
  var bar_stroke_width = '0.8px'
  var edit_stroke_width = '.25px'

  // when we want to plot the proportion in blood, find the max proportion
  // blood proportion is turned off right now
  var maxBlood = d3.max(nodes, function(x) {
    if (!x.children) {
      return +x.blood
    } else {
      return 0.0
    }
  })

  // find the minimum value in the data so that we can set a scale
  var absMin = 0.0
  var minProp = d3.min(nodes, function(x) {
    if (!x.children) {
      return Math.max(absMin,+x.max_organ_prop) * 100.0;
    } else {
      return 100.0
    }
  })
  var maxProp = d3.max(nodes, function(x) {
    if (!x.children) {
      return +x.max_organ_prop * 100.0;
    } else {
      return 0.0
    }
  })

  // we want the scale to go slightly lower than the minimum, so that the smallest value still has some minimal thickness
  var scaleCounts = d3.scale.log().domain([minProp,maxProp]).range([0.05,counts_bar_max_size])
  var xAxis = d3.svg.axis().scale(scaleCounts).orient("bottom").ticks(5).tickFormat(function (d) {
    return scaleCounts.tickFormat(4,d3.format(",2d"))(d)
  })
  var xAxis2 = d3.svg.axis().scale(scaleCounts).orient("top").ticks(5).tickFormat(function (d) {
    return scaleCounts.tickFormat(4,d3.format(",2d"))(d)
  })



  // what's the initial zoom level we should have? we want to scale it so the whole tree fits in view
  var initialScaleZoom = 2000.0 / Math.max(options.width,options.height);

    // zoom from https://bl.ocks.org/mbostock/6123708 and http://jsfiddle.net/LYuta/2/
  vis = d3.select(selector).append("svg:svg")
	.attr("width", "100%")
	.attr("height", "100%")
	.append("svg:g")
	.attr("viewBox", "0 0 " + boxWidth + " " + boxHeight) // this should be parameterized at some point
	.attr("transform", "translate(30, 30)scale(" + initialScaleZoom + "," + initialScaleZoom + ")")
	.append("g")
	.call(d3.behavior.zoom().scaleExtent([0, 10]).on("zoom", zoomed))
	.append("g");
    
    // make a rectangle that catches the zoom mouse movements; otherwise we can
  // only zoom over specific objects in the plot, which is really confusing to the user
    var catcher = vis.append("rect")
	.attr("width", "100%")
	.attr("height", "100%")
	.style("fill", "none")
	.style("pointer-events", "all");
    
    //respond to mouse, even when transparent
    // we need to reference vis in the function
    function zoomed() {
	vis.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }
    
    // put the axis on the plot at the top and the bottom
    vis.append("g")
	.attr("transform", "translate(" + (barplot_location) + "," + ( h + 2) + ")")
	.call(xAxis)
	.style("fill","none")
	.style("stroke","#000")
	.style("shape-rendering","crispEdges")
	.selectAll("text")
	.style("fill","black")
	.style("stroke-width",0)
	.style("shape-rendering","crispEdges")
    
    vis.append("g")
	.attr("transform", "translate(" + (barplot_location) + "," + ( -2 ) + ")")
	.call(xAxis2)
	.style("fill","none")
	.style("stroke","#000")
	.style("shape-rendering","crispEdges")
	.selectAll("text")
	.style("fill","black")
	.style("stroke-width",0)
	.style("shape-rendering","crispEdges")
    
    if (options.skipBranchLengthScaling) {
	var yscale = d3.scale.linear()
	    .domain([0, w])
	    .range([0, w]);
    } else {
	var yscale = scaleBranchLengths(nodes, w)
    }
    
    // add the links between nodes on the tree -- we use solid lines for normal nodes,
    // dashed lines for lines that lead to nodes that are duplicates when two organs have
    // the same allele
    var link = vis.selectAll("path.link")
	.data(tree.links(nodes))
	.enter().append("svg:path")
	.attr("class", "link")
	.attr("d", function(d,i) {
	    if (!d.target.is_spine){
		return diagonal(d,i)
	    } else {
		return caterpillar(d,i)
	    }
	})
	.attr("fill", "none")
	.style("stroke", function(d) {
	    if (d.target.is_spine){
		return "black"; // optionally set it to a different color to highlight caterpillar branches
	    } else {
		return "black";
	    }
	})
	.attr("stroke-width", function(d) {
	    if (!d.target.is_spine){
		return "1.4px";
	    } else {
		return "3px";
	    }
	})
    
	.style("stroke-dasharray", function(d) {
	    //if (!d.target.justOrganSplit || d.target.justOrganSplit == "false") {
	    if (!d.target.is_spine) { // remove if you don't want dashes for the caterpillar branches
		return ("0, 0");
	    } else {
		return ("1, 2");
	    }});
    
    // add all the non-tree data: reads, proportions, blood membership
    var drawNonTreeData = vis.selectAll("g.node")
	.data(nodes)
	.enter().append("rect")
	.filter(function(d) {return ! d.children })
	.each(function(nd) {
	    if (!nd.name.startsWith("internal")) {
		var event = nd.event// taxaToObj(d.name).event
		var eventArray = padWithMatches(hmidToEvents(nd.event),options.startRegion,options.endRegion)
		drawDottedConnector(nd)
		drawEditStrings(eventArray,nd, options.barWidth, options.endRegion-options.startRegion,barHeight)
		drawMembership(nd, barHeight,options.membershipBarWidth)
		drawCounts(nd,scaleCounts, barHeight)
	    }
	});
    
    // add a histogram for the proporiton of reads for this sample
    function drawCounts(d, scaleValue, barHeight) {
      var rectangle = vis.append("rect")
      .attr('fill', d.color) // taxaToObj(d.name).color)
      .attr('stroke', '#111')
      .attr('stroke-width', bar_stroke_width)
      .attr("width", barHeight)
      .attr("height", function(node) {
        // if the max organ proportion is less than 1% (and results in a NaN log score), set it to a minimum value
        return scaleValue(Math.max(absMin,+d.max_organ_prop) * 100.0)
      })
      .attr("transform","translate(" + barplot_location + "," + (d.x + barHeight/2.0) + ") rotate(-90) ");
    }

    // add a histogram for the proporiton of reads for this sample
    function drawAncestry(d, scaleValue, barHeight) {
      var rectangle = vis.append("rect")
      .attr('fill', d.color) // taxaToObj(d.name).color)
      .attr('stroke', '#111')
      .attr('stroke-width', bar_stroke_width)
      .attr("width", barHeight)
      .attr("height",scaleValue( d.max_organ_prop * 100.0 + 1.0)) // (taxaToObj(d.name).proportion // FIX
      .attr("transform","translate(" + barplot_location + "," + (d.x + barHeight/2.0) + ") rotate(-90) ");
    }

    // draw a dotted connector from the end of the tree branch to the nodes / reads
    function drawDottedConnector(d) {
      var myLine = vis.append("svg:line")
      .attr("class", 'd3-dp-line')
      .attr("y1", d.x)
      .attr("x1", d.y)
      .attr("y2", d.x)
      .attr("x2", event_location)
      .style("stroke-dasharray", ("2, 5"))
      .style("stroke-opacity", 1.5)
      .style("stroke", "gray")
    }

    // draw the nodes -- the boxes that represent the insertion and deletions over the read
    function drawEditStrings(eventArray, d, barLength,size, barHeight) {
      // scale from the event window to the barlength on the screen
      var scaleX = d3.scale.linear().range([0,barLength]).domain([0,size])
      var heatmap_colors = ['#FFFFFF','#CE343F','#2E4D8E','#D49E35'];

      for (i = 0; i < eventArray.length; i++) {
        var rectangle = vis.append("rect")
        .attr('fill', heatmap_colors[eventArray[i].typ] )
        .attr('id', d.event )
        .attr('stroke', '#111')
        .attr('stroke-width', edit_stroke_width)
        .attr("width", barHeight)
        .attr("height", function(d) {return scaleX(eventArray[i].len)})
        .attr("transform","translate(" + (event_location + scaleX(eventArray[i].pos)) + "," + (d.x + barHeight/2.0) + ") rotate(-90) ")
        .style('pointer-events','all')
        .on('mouseover', function(d) {
          //console.log(d3.select(this)[0][0].id)
        })
      }
    }

    // draw a box for membership -- which type of organ is this (by color)
    function drawMembership(d,barHeight,barWidth) {
      var rectangle = vis.append("rect")
      .attr('fill', d.color) // taxaToObj(d.name).color
      .attr('stroke', '#111')
      .attr('stroke-width', bar_stroke_width)
      .attr("width", barHeight)
      .attr("height", barWidth)
      .attr("transform","translate(" + membership_location + "," + (d.x + barHeight/2.0) + ") rotate(-90) ");
    }

    // draw the proportion seen in blood for this allele
    function drawBloodProp(d,barHeight,barWidth,maxBlood) {
      if (d.blood > 0.2) {
        var xxx = 0.0
      }

      var scaleX = d3.scale.linear().range([0,barWidth]).domain([0,maxBlood])
      clrs = ["#d9d9d9","#FF0000"];
      var rectangle = vis.append("rect")
      .attr('fill', clrs[1]) // clrs[taxaToObj(d.name).clade])
      .attr('stroke', '#111')
      .attr('stroke-width', bar_stroke_width)
      .attr("height", scaleX(d.blood))
      .attr("width", barHeight)
      .attr("transform","translate(" + primary_bar + "," + (d.x + barHeight/2.0) + ") rotate(-90) ");

      var rectangle2 = vis.append("rect")
      .attr('fill', clrs[0]) // clrs[taxaToObj(d.name).clade])
      .attr('stroke', '#111')
      .attr('stroke-width', bar_stroke_width)
      .attr("height", scaleX(maxBlood - d.blood))
      .attr("width", barHeight)
      .attr("transform","translate(" + (primary_bar + scaleX(d.blood)) + "," + (d.x + barHeight/2.0) + ") rotate(-90) ");
    }


    // ------------------------------------------------------------------------
    // draw the legend onto the plot
    // ------------------------------------------------------------------------
    function drawLegend(colorMap) {
      // remove any previous legends
      $( "#fixeddivleg" ).empty();

      var htmlOutput = "<b>Rectangle colors: </b><br>"
      $( '#fixeddivleg' ).html( htmlOutput );

      var maxPerColumn = 12

      // legend spacer between individual entries
      var legendSpace = 20
      var legendOffset = 10

      // make a new SVG for the legend and text
      var legend = d3.select("#fixeddivleg").append("svg")
      .attr("width", 300)
      .attr("height", 300)
      .append("svg:g")
      .attr("transform","translate(" + 20 + "," + 20 + ")");

      var rects = legend.selectAll("circle")
      .data(d3.entries(colorMap))
      .enter()
      .append("rect")
      .filter(function(d) {
        return d.key !== "UNKNOWN"
      })
      .attr("x", function(d, i) {
        if (i >= maxPerColumn)
        return(200)
        else
        return(30)
      })
      .attr("y", function(d, i) {
        if (i >= maxPerColumn)
        return (i - maxPerColumn) * legendSpace;
        else
        return (i) * legendSpace;
      })
      .attr("fill", function(d, i) {
        return(d.value);
      })
      .attr("width", 10)
      .attr("height", 10)


      legend.selectAll("text")
      .data(d3.entries(colorMap))
      .enter()
      .append("text")
      .filter(function(d) {
        return d.key !== "UNKNOWN"
      })
      .attr("x", function(d, i) {
        if (i >= maxPerColumn)
        return(200 + legendSpace)
        else
        return(30 + legendSpace)
      })
      .attr("y", function(d, i) {
        if (i >= maxPerColumn)
        return (i - maxPerColumn) * legendSpace + legendOffset;
        else
        return (i) * legendSpace + legendOffset;

      })
      .text(function(d,i){
        return d.key;
      })
    }


    // ------------------------------------------------------------------------
    // the circles for the end nodes
    // ------------------------------------------------------------------------

    var pie = d3.layout.pie()
    .sort(null)
    .value(function(d) {
      return d.cladeProportions;
    });

    var drawNodeCircles = vis.selectAll("g.node.myGroup")
    .data(nodes)
    .enter()
    .append("g");

    var cirColor = d3.scale.category20();

    drawNodeCircles.each(function (d, i) {
	selection = d3.select(this);
	
      var circle = selection.append("circle")
      .attr("r", function(d) {
        if (d.parent == "null") {
          return circleSize * 3;
        } else {
          return circleSize;
        }
      })
      .attr('fill', function(d) {
        if (d.nodecolor) {
          return d.nodecolor
        } else {
          return "black"
        }
      })
      .attr('stroke', circleOutline)
      .attr('stroke-width', 0.5)
      .attr("cy", function(d) {
        return d.x
      })
      .attr("cx", function(d) {
        return d.y
      });
    });

    // ------------------------------------------------------------------------
    // given a node object name, get the organ proportion under this node
    // ------------------------------------------------------------------------
    function drawHoverBarChart(d, xPos, yPos, proposedHeight, proposedWidth) {

      // add a box of text about this node
      var htmlOutput = "<b>Node name: </b> " + d.name + "<br><br><b>depth: </b>" + d.depth
      htmlOutput += "<br><br><b>Edits shared by child nodes (* for not shared):</b><br>" + d.commonEvent.split("_").join(", ")
      htmlOutput += "<br><br><b>Allele (inferred for internal nodes):</b><br>" + d.event.split("_").join(", ") + "<br><br><b>Proportion of sampled organs / organism:</b><br>"
      $( '#fixeddivinfo' ).html( htmlOutput );

      var margin = {top: 10, right: 10, bottom: 100, left: 60}
      var height = proposedHeight - margin.top - margin.bottom;
      var width = proposedWidth - margin.left - margin.right;

      //var x = d3.scale.ordinal().domain(d3.keys(d.organCounts)).rangeRoundBands([0, width], .15);
      var x = d3.scale.ordinal().domain(d3.keys(d.organProportions)).rangeRoundBands([0, width], .15);


      var y = d3.scale.linear().domain([0.0,1.0]).range([height, 0]);
      //var y = d3.scale.linear().domain([0.0,d.organCountsMax]).range([height, 0]);

      var yAxis = d3.svg.axis().scale(y).orient("left")
      var xAxis = d3.svg.axis()
      .scale(x)
      .orient("bottom");

      var assoc = {}
      //$.map(d.organCounts, function(value,key) {
      $.map(d.organProportions, function(value,key) {
        assoc[key] = value;
      })

      var vis2 = d3.select("#fixeddivinfo").append("svg:svg")
      .attr("width", 320)
      .attr("height", 500)
      .append("svg:g")
      .attr("transform","translate(" + 50 + "," + 20 + ")");

      vis2.append("g")
      .attr("class", "y axis")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")

      vis2.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(" + 0 + "," + height + ")")
      .call(xAxis)
      .selectAll("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", "-.5em")
      .attr("transform", function(d) {
        return "rotate(-90)"
      });

      // create the barplot for each node in the tree
      vis2.selectAll(".hoverBarPlot")
      .data(d3.entries(assoc))
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", function(d,i) {
        return x(d.key);
      })
      .attr("width", x.rangeBand())
      .attr("y", function(d,i) {
        return y(d.value);
      })
      .attr("height", function(d) {
        return height - y(d.value);
      })
      .style("fill", function(d,i) {
        return "gray"; // return newColorMap2[d.key];
      })
      .style("stroke", "black");


    }

    // ------------------------------------------------------------------------
    // return our tree and visualization
    // ------------------------------------------------------------------------

    return {tree: tree, vis: vis}
  }

  var match = 0
  var insertion = 2
  var deletion = 1


  // ********************************************************************************************************
  // process the encoded GESTALT strings into a series of object for D3 to draw on the screen
  // ********************************************************************************************************
  function padWithMatches(eventObjectArray, start, end) {
    var resultsArray = [];
    for (i = 0; i < eventObjectArray.length; i++) {
      var curEvt = offsetByStart(eventObjectArray[i],start)

      // we haven't added an event yet, pad to the beginning of the target region with a match
      if (resultsArray.length  == 0) {
        if (curEvt.pos > 0) { // we've offset by the start, so the beginning is now zero
        resultsArray.push({typ: match, len: curEvt.pos, pos: 0})
      }
      resultsArray.push(curEvt)
    }
    // we have values in the array, pad matches between events
    else {
      if (curEvt.pos > resultsArray[resultsArray.length -1].pos + resultsArray[resultsArray.length -1].len) {
        resultsArray.push({typ: match, len: curEvt.pos - (resultsArray[resultsArray.length -1].pos + resultsArray[resultsArray.length -1].len), pos: resultsArray[resultsArray.length -1].pos + resultsArray[resultsArray.length -1].len})
      }
      resultsArray.push(curEvt)
    }
  }
  // now pad the end -- if the last event doesn't run all the way to the end pos, add matches
  if (resultsArray.length == 0) {
    resultsArray.push({typ: match, len: (end - start), pos: 0})
  }
  if (resultsArray[resultsArray.length - 1].pos < (end - start)) {
    resultsArray.push({typ: match, len: (end - start)  - (resultsArray[resultsArray.length -1].pos + resultsArray[resultsArray.length -1].len), pos: resultsArray[resultsArray.length -1].pos + resultsArray[resultsArray.length -1].len})
  }
  return resultsArray;
}

// transform the HMID event by subtracting the start location of the HMID region so we have events that start at zero
function offsetByStart(event,start) {
  return ({typ: event.typ, len: event.len, pos: event.pos - start});
}

// take an HMID event string and convert to a series of events
function hmidToEvents(eventString) {
  var res = eventString.split("_");
  var tokens = [];
  for (i = 0; i < res.length; i++) {
    var subEvents = res[i].split("&")

    for (j = 0; j < subEvents.length; j++)
    if (subEvents[j] != "NONE") {
      tokens.push(eventToObject(subEvents[j]))
    }
  }
  return tokens;
}

// convert a single event string to an object, which looks like
function eventToObject(event) {
  var tokens = event.split("+");
  var typeOf = tokens[0].substring(tokens[0].length - 1, tokens[0].length)
  if (typeOf == "I") {
    typeOf = insertion
  } else if (typeOf == "D") {
    typeOf = deletion
  } else {
    typeOf = match
  }

  var lengthOf = Number(tokens[0].substring(0, tokens[0].length - 1))
  var position = Number(tokens[1])
  return {typ: typeOf, len: lengthOf, pos: position}
}

function drawEventTooltip(d) {
  var div = d3.select("body")
  .append("div")  // declare the tooltip div
  .attr("class", ".tooltip")              // apply the 'tooltip' class
  .style("left", (d3.event.pageX + 10) + "px")
  .style("top", (d3.event.pageY - 18) + "px")
  .style("width", "300px")
  .style("height", "50px")
  .style("stroke", "black")
  .style("opacity", .9);
}


function flash(name, dy) {
  return function() {
    d3.select(this).append("text")
    .attr("class", name)
    .attr("transform", "translate(" + d3.mouse(this) + ")")
    .attr("dy", dy)
    .text(name)
    .transition()
    .duration(1500)
    .style("opacity", 0)
    .remove();
  };
}

function legendColor(rootNode) {
  organToColor = []
  legendToColorRec(rootNode,organToColor)
  return(organToColor)
}

function legendToColorRec(rootNode,colorMap) {
  colorMap[rootNode.sample] = rootNode.color
  if (rootNode && rootNode.children) {
    for (var i = 0, l = rootNode.children.length; i < l; ++i) {
      legendToColorRec(rootNode.children[i],organToColor)
    }
  }
  return(colorMap);
}

// ************************************************************************************************************************
//
// The main entry point for the script -- populate the list of trees, and load and draw the tree once the user selects one
//
// ************************************************************************************************************************


// ----------------------------------------------------------------------
// datasets
// ----------------------------------------------------------------------

function load() {
  // First clean up the canvas
  d3.select(selector).selectAll("*").remove();

  //load the external data -- cell_culture_parsimony_tree.json
  d3.json(current_data.tree_file , function(error, treeData) {
    newick = treeData;

    // ------------------------------------------------------------------------------------------------------
    // the main function call to build the phylogeny
    build_tree(selector, newick[0], current_data);
  });
}
var dims = {width: 2000,height: 1000}
var selector = '#phylogram'
var vis = d3.select(selector).append("svg:svg")
.attr("width", dims.width + 600)
.attr("height", dims.height + 200)
.append("svg:g")
.attr("transform", "translate(20, 20)")

boxWidth = 800
boxHeight = 800
sidebarWidth = 200
sidebarHeight = 800

// populate the tree list
var allTrees = ""
$.ajax({
  url: 'tree_master_list.json',
  async: false,
  dataType: 'json',
  success: function (response) {
    allTrees = response
    var $choices = $("#dataSelect");
    $choices.empty();
    $.each(response, function(index, value) {
      $('#dataSelect')
      .append($("<option></option>")
      .attr("value",value.tree_file)
      .text(value.name));
    });
  },
  error: function(XMLHttpRequest, textStatus, errorThrown) {
    alert("Status: " + textStatus); alert("Error: " + errorThrown);
  }
});

function updateData() {
  var optionSelected = $("#dataSelect").val();
  allTrees.forEach(function(entry) {
    if (entry.tree_file == optionSelected)
    current_data = entry
  })
  load()
}

/** load up the jquery stuff **/
$( document ).on( "pageinit", "#demo-page", function() {
  $( document ).on( "swipeleft swiperight", "#demo-page", function( e ) {
    // We check if there is no open panel on the page because otherwise
    // a swipe to close the left panel would also open the right panel (and v.v.).
    // We do this by checking the data that the framework stores on the page element (panel: open).
    if ( $.mobile.activePage.jqmData( "panel" ) !== "open" ) {
      if ( e.type === "swipeleft"  ) {
        $( "#mypanel" ).panel( "open" );
      } else if ( e.type === "swiperight" ) {
        $( "#mypanel" ).panel( "open" );
      }
    }
  });
});
