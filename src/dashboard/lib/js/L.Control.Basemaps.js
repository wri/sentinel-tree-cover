L.Control.Basemaps = L.Control.extend({
    _map: null,
    includes: L.Evented ? L.Evented.prototype: L.Mixin.Event,
    options: {
        position: 'bottomright',
        tileX: 0,
        tileY: 0,
        tileZ: 0,
        layers: []  // list of basemap layer objects, first in list is default and added to map with this control
    },
    basemap: null,
    onAdd: function (map) {
        this._map = map;
        var container = L.DomUtil.create('div', 'basemaps leaflet-control closed');

        // disable events
        L.DomEvent.disableClickPropagation(container);
        if (!L.Browser.touch) {
            L.DomEvent.disableScrollPropagation(container);
        }

        this.options.basemaps.forEach(function(d, i){
            var basemapClass = 'basemap';

            if (i === 0) {
                this.basemap = d;
                this._map.addLayer(d);
                basemapClass += ' active';
            }
            else if (i === 1) {
                basemapClass += ' alt'
            }

            if (d.options.iconURL) {
                url = d.options.iconURL;
                console.log('url is: ', url);
            }
            else {
                var coords = {x: this.options.tileX, y: this.options.tileY};
                var url = L.Util.template(d._url, L.extend({
                    s: d._getSubdomain(coords),
                    x: coords.x,
                    y: d.options.tms ? d._globalTileRange.max.y - coords.y : coords.y,
                    z: this.options.tileZ
                }, d.options));

                if (d instanceof L.TileLayer.WMS) {
                    // d may not yet be initialized, yet functions below expect ._map to be set
                    d._map = map;

                    // unfortunately, calling d.getTileUrl() does not work due to scope issues
                    // have to replicate some of the logic from L.TileLayer.WMS

                    // adapted from L.TileLayer.WMS::onAdd
                    var crs = d.options.crs || map.options.crs;
                    var wmsParams = L.extend({}, d.wmsParams);
                    var wmsVersion = parseFloat(wmsParams.version);
                    var projectionKey = wmsVersion >= 1.3 ? 'crs' : 'srs';
                    wmsParams[projectionKey] = crs.code;

                    // adapted from L.TileLayer.WMS::getTileUrl
                    var coords2 = L.point(coords);
                    coords2.z = this.options.tileZ;
                    var tileBounds = d._tileCoordsToBounds(coords2);
                    var nw = crs.project(tileBounds.getNorthWest());
                    var se = crs.project(tileBounds.getSouthEast());
                    var bbox = (wmsVersion >= 1.3 && crs === L.CRS.EPSG4326 ?
                        [se.y, nw.x, nw.y, se.x] :
                        [nw.x, se.y, se.x, nw.y]).join(',');

                    url += L.Util.getParamString(wmsParams, url, d.options.uppercase) +
                        (d.options.uppercase ? '&BBOX=' : '&bbox=') + bbox;
                }
            }

            var basemapNode = L.DomUtil.create('div', basemapClass, container);
            var imgNode = L.DomUtil.create('img', null, basemapNode);
            imgNode.src = url;
            if (d.options && d.options.label) {
                imgNode.title = d.options.label;
            }

            L.DomEvent.on(basemapNode, 'click', function() {
                console.log(d);
                console.log(i);
                //console.log(l);
                console.log(this.basemap);
                //if different, remove previous basemap, and add new one
                if (d != this.basemap) {
                    map.removeLayer(this.basemap);
                    map.addLayer(d);
                    d.bringToBack();
                    map.fire('baselayerchange', d);
                    this.basemap = d;

                    L.DomUtil.removeClass(document.getElementsByClassName('basemap active')[0], 'active');
                    L.DomUtil.addClass(basemapNode, 'active');

                    var altIdx = (i+1) % this.options.basemaps.length;
                    L.DomUtil.removeClass(document.getElementsByClassName('basemap alt')[0], 'alt');
                    L.DomUtil.addClass(document.getElementsByClassName('basemap')[altIdx], 'alt');
                }
            }, this);

        }, this);

        if (this.options.basemaps.length > 2) {
            L.DomEvent.on(container, 'mouseenter', function () {
                L.DomUtil.removeClass(container, 'closed');
            }, this);

            L.DomEvent.on(container, 'mouseleave', function () {
                L.DomUtil.addClass(container, 'closed');
            }, this);
        }

        this._container = container;
        return this._container;
    }
});

L.Control.Basemaps2 = L.Control.extend({
    _map: null,
    includes: L.Evented ? L.Evented.prototype: L.Mixin.Event,
    options: {
        position: 'bottomright',
        tileX: 0,
        tileY: 0,
        tileZ: 0,
        layers: []  // list of basemap layer objects, first in list is default and added to map with this control
    },
    basemap: null,
    onAdd: function (map) {
        m1 = map;
        var l = 0;
        var container = L.DomUtil.create('div', 'basemaps leaflet-control closed');

        // disable events
        L.DomEvent.disableClickPropagation(container);
        if (!L.Browser.touch) {
            L.DomEvent.disableScrollPropagation(container);
        }

        this.options.basemaps.forEach(function(m, l){
            var basemapClass1 = 'basemap';

            if (l === 0) {
                this.basemap = m;
                m1.addLayer(m);
                basemapClass1 += ' active 1';
            }
            else if (l === 1) {
                basemapClass1 += ' alt 1'
            }

            if (m.options.iconURL) {
                url = m.options.iconURL;
                console.log('url is: ', url);
            }
            else {
                var coords = {x: this.options.tileX, y: this.options.tileY};
                var url = L.Util.template(m._url, L.extend({
                    s: m._getSubdomain(coords),
                    x: coords.x,
                    y: m.options.tms ? m._globalTileRange.max.y - coords.y : coords.y,
                    z: this.options.tileZ
                }, m.options));

                if (m instanceof L.TileLayer.WMS) {
                    // d may not yet be initialized, yet functions below expect ._map to be set
                    m1 = map;

                    // unfortunately, calling d.getTileUrl() does not work due to scope issues
                    // have to replicate some of the logic from L.TileLayer.WMS

                    // adapted from L.TileLayer.WMS::onAdd
                    var crs = m.options.crs || map.options.crs;
                    var wmsParams = L.extend({}, m.wmsParams);
                    var wmsVersion = parseFloat(wmsParams.version);
                    var projectionKey = wmsVersion >= 1.3 ? 'crs' : 'srs';
                    wmsParams[projectionKey] = crs.code;

                    // adapted from L.TileLayer.WMS::getTileUrl
                    var coords2 = L.point(coords);
                    coords2.z = this.options.tileZ;
                    var tileBounds = m._tileCoordsToBounds(coords2);
                    var nw = crs.project(tileBounds.getNorthWest());
                    var se = crs.project(tileBounds.getSouthEast());
                    var bbox = (wmsVersion >= 1.3 && crs === L.CRS.EPSG4326 ?
                        [se.y, nw.x, nw.y, se.x] :
                        [nw.x, se.y, se.x, nw.y]).join(',');

                    url += L.Util.getParamString(wmsParams, url, m.options.uppercase) +
                        (m.options.uppercase ? '&BBOX=' : '&bbox=') + bbox;
                }
            }

            var basemapNode = L.DomUtil.create('div', basemapClass1, container);
            var imgNode = L.DomUtil.create('img', null, basemapNode);
            imgNode.src = url;
            if (m.options && m.options.label) {
                imgNode.title = m.options.label;
            }
            iter = 2;
            var bm1 = L.tileLayer('https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
    {minZoom:8,
    attribution: '<a href="http://datadriven.yale.edu">Data-Driven Lab</a> | <a href="https://www.socialconnectedness.org"> Samuel Centre for Social Connectedness</a>'
  });

            var bm2 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: ' | Tiles &copy; Esri & Partners',
  minZoom:8
  })
            L.DomEvent.on(basemapNode, 'click', function() {
                //if different, remove previous basemap, and add new one
                console.log(m);
                console.log(l);
                console.log(iter);
                //console.log(l);
                console.log(this.basemap);
                if (iter % 2 == 0) {
                    m1.removeLayer(bm2);
                    m1.removeLayer(m);
                    bm2.addTo(m1);
                    //m.bringToBack();
                    map.fire('baselayerchange', bm2);
                    //this.basemap = m;

                    L.DomUtil.removeClass(document.getElementsByClassName('basemap active 1')[0], 'active');
                    L.DomUtil.addClass(basemapNode, 'active');

                    var altIdx1 = (l+1) % this.options.basemaps.length;
                    L.DomUtil.removeClass(document.getElementsByClassName('basemap alt 1')[0], 'alt1');
                    L.DomUtil.addClass(document.getElementsByClassName('basemap')[altIdx1], 'alt1');
                    iter += 1
                } else if (iter % 2 != 0) {
                    m1.removeLayer(m);
                    m1.removeLayer(bm2)
                    bm1.addTo(m1);
                    console.log("this was ran");
                    //m.bringToBack();
                    map.fire('baselayerchange', bm1);
                    //this.basemap = m;

                    L.DomUtil.removeClass(document.getElementsByClassName('basemap active 1')[0], 'active1');
                    L.DomUtil.addClass(basemapNode, 'active');

                    var altIdx1 = (l+1) % this.options.basemaps.length;
                    L.DomUtil.removeClass(document.getElementsByClassName('basemap alt 1')[0], 'alt1');
                    L.DomUtil.addClass(document.getElementsByClassName('basemap')[altIdx1], 'alt1');
                    iter += 1
                }
            }, this);

        }, this);

        if (this.options.basemaps.length > 2) {
            L.DomEvent.on(container, 'mouseenter', function () {
                L.DomUtil.removeClass(container, 'closed');
            }, this);

            L.DomEvent.on(container, 'mouseleave', function () {
                L.DomUtil.addClass(container, 'closed');
            }, this);
        }

        this._container = container;
        return this._container;
    }
});

L.control.basemaps = function (options) {
  return new L.Control.Basemaps(options);
};

L.control.basemaps2 = function (options) {
  return new L.Control.Basemaps2(options);
};
