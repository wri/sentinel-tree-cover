
/* ########################################
####### FUNCTIONALITY FOR NAVBAR ##########
######################################## */

/* SEARCHABLE CITY SELECT FUNCTIONALITY */

/* TO ADD A NEW INDICATOR */
/* 1. Update the HTML for index and splitscr */
/* 2. create a getColorIndicator function */
/* 3. create Indicatorstyle function */
/* 4. create Indicatorlegend function */
/* 5. create Indicatorstyle_set function */
/* 6. update changecity function */
/* 7. update removelegends function */
/* 8. Update set_current_style function */

$('.ui.dropdown')
    .dropdown()
    ;

/* ################### *
// m.setMaxBounds([[38.19, 114.42], [42.19, 118.42]]).setView([40.19, 116.42], 9),
/* The following functions give functionality to the navbar and to the indicator selectors*/

function applyMargins() {
    var leftToggler = $(".mini-submenu-left");
    if (leftToggler.is(":visible")) {
        $("#map .ol-zoom")
            .css("margin-left", 0)
            .removeClass("zoom-top-opened-sidebar")
            .addClass("zoom-top-collapsed");
    } else {
        $("#map .ol-zoom")
            .css("margin-left", $(".sidebar-left").width())
            .removeClass("zoom-top-opened-sidebar")
            .removeClass("zoom-top-collapsed");
    }
}

function isConstrained() {
    return $(".sidebar").width() == $(window).width();
}
function applyInitialUIState() {
    if (isConstrained()) {
        $(".sidebar-left .sidebar-body").fadeOut('slide');
        $('.mini-submenu-left').fadeIn();
    }
}

$(function () {
    $('.sidebar-left .slide-submenu').on('click', function () {
        var thisEl = $(this);
        thisEl.closest('.sidebar-body').fadeOut('slide', function () {
            $('.mini-submenu-left').fadeIn();
            applyMargins();
        });
    });
    $('.mini-submenu-left').on('click', function () {
        var thisEl = $(this);
        $('.sidebar-left .sidebar-body').toggle('slide');
        thisEl.hide();
        applyMargins();
    });
    $(window).on("resize", applyMargins);


    /// This function makes it so that when a user clicks outside of the popover it closes
    $(document).on('click', function (e) {
        $('[data-toggle="popover"],[data-original-title]').each(function () {
            //the 'is' for buttons that trigger popups
            //the 'has' for icons within a button that triggers a popup
            if (!$(this).is(e.target) && $(this).has(e.target).length === 0 && $('.popover').has(e.target).length === 0) {
                (($(this).popover('hide').data('bs.popover') || {}).inState || {}).click = false  // fix for BS 3.3.6
            }

        });
    });

    /* ####################################
    ####### BEGIN MAP JAVASCRIPT ##########
    ####################################### */


    /* ###################################
    ### USE LOCAL STORAGE TO SYNC ########
    ### MAP VIEW BETWEEN PAGES    ########
    ### DELETE LOCAL STORAGE WHEN ########
    ### DONE                      ########
    ################################### */

    var current_indicator = "income";

    var m = L.map('map', { zoomControl: false })
    //m.setMaxBounds[(13.54, 38.155), (13.58, 38.159)]
    m.setView([13.58941, 38.19342], 15)
    //m.setMaxBounds[(13.5, 38.15), (13.6, 38.16)]

    /* ###################################
    ### INSTANTIATE MAP AND SET   ########
    ### COORDINATES. CREATE BASE  ########
    ### MAP SWITCHER              ########
    ################################### */

    m.createPane('left')
    m.createPane('right')

    var myLayer1 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '<a href="http://datadriven.yale.edu">Data-Driven Lab</a> | <a href="https://www.socialconnectedness.org"> Samuel Centre for Social Connectedness</a> | Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            minZoom: 14, maxZoom: 17}).addTo(m);

    var myLayer2 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '<a href="http://datadriven.yale.edu">Data-Driven Lab</a> | <a href="https://www.socialconnectedness.org"> Samuel Centre for Social Connectedness</a> | Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            minZoom: 14, maxZoom: 17, pane: 'right'}).addTo(m);


    var treecover_raw = L.leafletGeotiff(url = "includes/tigray_leaflet2.tif",
                                         options = { band: 0, displayMin: 0.01, displayMax: 1, colorScale: 'viridis', clampLow: false, clampHigh: true, pane: "right"})

    treecover_raw.addTo(m);
    
    L.control.sideBySide(myLayer1, [myLayer2, treecover_raw]).addTo(m);


    new L.Control.Zoom({ position: 'topright' }).addTo(m);
    new L.control.scale().addTo(m);
   
    m.addLayer(treecover_raw)

    var overlayMaps = {
        "Data": treecover_raw
    };

    L.control.layers(null, overlayMaps).addTo(m);

    //m.doubleClickZoom.disable();
    //Highlight on mouse over
    function highlightFeature(e) {
        var layer = e.target;
        layer.setStyle({
            weight: 5,
            color: 'white',
            dashArray: '',
            fillOpacity: 0.2
        });
        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
            layer.bringToFront();
        }
    }

    //Remove highlight on mouse leave
    function resetHighlight(e) {
        var layer = e.target;
        layer.setStyle({
            weight: 1,
            color: 'black',
            dashArray: '',
            opacity: 1,
            fillOpacity: 0.6
        });
    }

    /* ###############################
    ####### ON EACH LAYER  ###########
    ################################## */

    //creates a pop up layer showing name, income, and population
    //tells the geojson to highlight upon mouse over
    function onEachFeature(feature, layer) {
        pzoom = m.getZoom();
        layer.on({
            mouseover: highlightFeature,
            click: zoomToFeature,
            mouseout: resetHighlight
        })
    }
})

$(document).ready(function () {
    //incstyle();
    $('.ui.dropdown')
        .dropdown()
        ;
    $('[data-toggle="tooltip"]').tooltip();
    $('[data-toggle="popover"]').popover();
});
