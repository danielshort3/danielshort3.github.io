"""Build a native Pizza Delivery Tableau package from its published snapshot.

Run: python build/tableau/rebuild-pizza.py
This edits workbook XML only. The original Hyper, logical relationships, and
published workbook/dashboard identities are preserved. Native Tableau opening
and filter/render verification are required before this artifact is published.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
import zipfile


REPO = Path(__file__).resolve().parents[2]
USER_NS = "http://www.tableausoftware.com/xml/user"
ET.register_namespace("user", USER_NS)
BLUE = "#005fed"
NAVY = "#091f3b"
SLATE = "#475569"
PALE = "#d8e9ff"
MIST = "#eef2f7"
BG = "#f9f9fa"
MAIN = "Pizza Delivery Dashboard"
MONEY = 'c"$"#,##0.00;("$"#,##0.00)'
INTEGER = "#,##0"


def elem(parent, tag, text=None, **attrs):
  node = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
  if text is not None:
    node.text = text
  return node


def uid(label):
  return "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, "https://www.danielshort.me/tableau/pizza/" + label)).upper() + "}"


def fmt(rule, attr, value, **attrs):
  return elem(rule, "format", attr=attr, value=value, **attrs)


def style_rule(parent, element, **formats):
  rule = elem(parent, "style-rule", element=element)
  for attr, value in formats.items():
    fmt(rule, attr.replace("_", "-"), value)
  return rule


def plain_style(table):
  style = elem(table, "style")
  style_rule(style, "worksheet", font_family="Arial", font_size="11", color=NAVY,
             display_field_labels="false")
  style_rule(style, "table", background_color="#ffffff")
  style_rule(style, "cell", color=NAVY, font_family="Arial", font_size="11")
  style_rule(style, "header", color=NAVY, font_family="Arial", font_size="11")
  style_rule(style, "gridline", stroke_size="1", stroke_color=MIST)
  style_rule(style, "zeroline", stroke_size="1", stroke_color="#cbd5e1")
  style_rule(style, "table-div", stroke_size="0")
  style_rule(style, "header-div", stroke_size="0")
  return style


class PizzaBuilder:
  def __init__(self, root):
    self.root = root
    self.source = root.find("./datasources/datasource")
    self.ds = self.source.get("name")
    self.cols = {x.get("name"): copy.deepcopy(x) for x in self.source.findall("column")}
    self.old_dashboard = copy.deepcopy(root.find("./dashboards/dashboard"))
    self.original_connections = ET.tostring(self.source.find("connection"))
    self.original_objects = ET.tostring(self.source.find("object-graph"))
    self.sheets = []
    self.action = "[Action_Pizza_City_" + uuid.uuid5(uuid.NAMESPACE_URL, "pizza-city-select").hex.upper() + "]"
    self.date_ref = "[none:Date:qk]"
    self.instances = {}
    self.city = self.instance("City", "None", "none", "nominal", "nk")
    self.housing = self.instance("Housing", "None", "none", "nominal", "nk")
    self.date = self.instance("Date", "None", "none", "quantitative", "qk")
    self.month = self.instance("Date", "Month-Trunc", "tmn", "quantitative", "qk")
    self.tip = self.instance("Tip", "Avg", "avg", "quantitative", "qk")
    self.median = self.instance("Tip", "Median", "med", "quantitative", "qk")
    self.total = self.instance("Tip", "Sum", "sum", "quantitative", "qk")
    self.count = self.instance("Tip", "Count", "cnt", "quantitative", "qk")
    self.minutes = self.instance("Calculation_1753589172811726849", "Avg", "avg", "quantitative", "qk")
    self.tip_bin = self.instance("Tip (bin)", "None", "none", "quantitative", "qk")
    self.time_bin = self.instance("Delivery_Length (bin)", "None", "none", "quantitative", "qk")
    self.sparse = self.calculated("Pizza sample size", "IF COUNT([Tip]) < 30 THEN 'Fewer than 30' ELSE '30 or more' END", "string", "nominal", "nk")
    self.count_label = self.calculated("Pizza deliveries label", "STR(COUNT([Tip])) + IF COUNT([Tip]) < 30 THEN '*' ELSE '' END", "string", "nominal", "nk")
    self.selected_median = self.calculated("Pizza selected median", "{ EXCLUDE [Tip (bin)] : MEDIAN([Tip]) }", "real", "quantitative", "qk", derivation="Avg", prefix="avg")

  def qualify(self, field):
    return f"[{self.ds}].{field}"

  def instance(self, field, derivation, prefix, dtype, suffix):
    name = f"[{prefix}:{field}:{suffix}]"
    self.instances[name] = ET.Element("column-instance", {
      "column": f"[{field}]", "derivation": derivation, "name": name,
      "pivot": "key", "type": dtype,
    })
    return name

  def calculated(self, caption, formula, datatype, dtype, suffix, derivation="User", prefix="usr"):
    field = "Calculation_Pizza_" + caption.replace(" ", "_")
    col = ET.Element("column", {"name": f"[{field}]", "caption": caption,
      "datatype": datatype, "role": "measure", "type": dtype})
    elem(col, "calculation", **{"class": "tableau", "formula": formula})
    self.cols[col.get("name")] = col
    insertion = next((i for i, x in enumerate(self.source) if x.tag == "column"), len(self.source))
    self.source.insert(insertion, copy.deepcopy(col))
    return self.instance(field, derivation, prefix, dtype, suffix)

  def add_filters(self, view, city_action=True):
    for group, field in ((101, self.city), (102, self.housing)):
      filt = elem(view, "filter", **{"class": "categorical", "column": self.qualify(field), "filter-group": str(group)})
      elem(filt, "groupfilter", function="level-members", level=field,
        **{f"{{{USER_NS}}}ui-enumeration": "all", f"{{{USER_NS}}}ui-marker": "enumerate"})
    filt = elem(view, "filter", **{"class": "quantitative", "column": self.qualify(self.date),
      "filter-group": "103", "included-values": "in-range"})
    elem(filt, "min", "#2017-07-05#")
    elem(filt, "max", "#2018-09-23#")
    if city_action:
      filt = elem(view, "filter", **{"class": "categorical", "column": self.qualify("[Action (City)]")})
      elem(filt, "groupfilter", function="level-members", level="[City]",
        **{f"{{{USER_NS}}}ui-action-filter": self.action,
          f"{{{USER_NS}}}ui-enumeration": "all", f"{{{USER_NS}}}ui-marker": "enumerate"})

  def sheet(self, name, rows=None, cols=None, mark="Bar", labels=(), tooltip=(),
            city_sort=False, show_city=True, color=BLUE, label_size=11, action=True):
    sheet = ET.Element("worksheet", {"name": name})
    layout = elem(sheet, "layout-options")
    elem(elem(layout, "title"), "formatted-text")
    table = elem(sheet, "table")
    view = elem(table, "view")
    sources = elem(view, "datasources")
    elem(sources, "datasource", caption=self.source.get("caption"), name=self.ds)
    deps = elem(view, "datasource-dependencies", datasource=self.ds)
    # Keep only delivery columns; no weather/ZIP values enter any displayed metric.
    wanted = {"[City]", "[Date]", "[Housing]", "[Tip]", "[Tip (bin)]",
      "[Calculation_1753589172811726849]", "[Total Delivery Time]",
      "[Total Delivery Time - Split 1]", "[Total Delivery Time - Split 2]",
      "[Delivery_Length (bin)]"}
    wanted.update(x.get("column") for x in self.instances.values())
    for field in sorted(wanted):
      if field in self.cols:
        deps.append(copy.deepcopy(self.cols[field]))
    for instance in self.instances.values():
      deps.append(copy.deepcopy(instance))
    self.add_filters(view, action)
    if city_sort:
      elem(view, "computed-sort", column=self.qualify(self.city), direction="DESC", using=self.qualify(self.count))
    slices = elem(view, "slices")
    for field in (self.city, self.housing, self.date):
      elem(slices, "column", self.qualify(field))
    if action:
      elem(slices, "column", self.qualify("[Action (City)]"))
    elem(view, "aggregation", value="true")
    style = plain_style(table)
    cell = style_rule(style, "cell")
    for field in (self.tip, self.total, self.median, self.selected_median):
      fmt(cell, "text-format", MONEY, field=self.qualify(field))
    fmt(cell, "text-format", INTEGER, field=self.qualify(self.count))
    fmt(cell, "text-format", "0.0", field=self.qualify(self.minutes))
    if city_sort:
      fmt(cell, "height", "33", field=self.qualify(self.city))
      header = style_rule(style, "header")
      fmt(header, "width", "118", field=self.qualify(self.city))
      if not show_city:
        fmt(header, "display", "false", field=self.qualify(self.city))
    pane = elem(elem(table, "panes"), "pane", selection_relaxation_option="selection-relaxation-allow")
    elem(elem(pane, "view"), "breakdown", value="auto")
    elem(pane, "mark", **{"class": mark})
    enc = elem(pane, "encodings")
    for field in labels:
      elem(enc, "text", column=self.qualify(field))
    for field in tooltip:
      elem(enc, "tooltip", column=self.qualify(field))
    mark_style = style_rule(elem(pane, "style"), "mark", mark_color=color,
      mark_labels_show="true" if labels else "false", mark_labels_cull="false", font_size=str(label_size),
      font_family="Arial", color=NAVY)
    if mark == "Text":
      fmt(mark_style, "mark-labels-show", "true")
      style_rule(style, "cell", text_align="left", vertical_align="center")
    elem(table, "rows", self.qualify(rows) if rows else None)
    elem(table, "cols", self.qualify(cols) if cols else None)
    elem(sheet, "simple-id", uuid=uid(name))
    self.sheets.append(sheet)
    return sheet

  def axis(self, sheet, field, scope, title=None, minimum=0, maximum=None, visible=True):
    style = sheet.find("table/style")
    rule = style_rule(style, "axis")
    ref = self.qualify(field)
    fmt(rule, "display", str(visible).lower(), field=ref, scope=scope, **{"class": "0"})
    if title is not None:
      fmt(rule, "title", title, field=ref, scope=scope, **{"class": "0"})
    if maximum is not None:
      enc = elem(rule, "encoding", attr="space", **{"class": "0", "field": ref,
        "field-type": "quantitative", "scope": scope, "type": "space", "range-type": "fixed",
        "min": minimum, "max": maximum})
    elif scope == "rows":
      elem(rule, "encoding", attr="space", **{"class": "0", "field": ref,
        "field-type": "quantitative", "scope": scope, "type": "space", "range-type": "fixedmin", "min": minimum})
    else:
      elem(rule, "encoding", attr="space", **{"class": "0", "field": ref,
        "field-type": "quantitative", "scope": scope, "type": "space", "domain-expand": "true"})

  def build_sheets(self):
    self.sheet("Pizza - Deliveries", mark="Text", labels=(self.count,), label_size=28)
    self.sheet("Pizza - Median tip", mark="Text", labels=(self.median,), label_size=28)
    self.sheet("Pizza - Total tips", mark="Text", labels=(self.total,), label_size=28)
    city = self.sheet("Pizza - City average tip", rows=self.city, cols=self.tip, labels=(self.tip,),
      tooltip=(self.count, self.minutes, self.sparse), city_sort=True, action=False)
    self.axis(city, self.tip, "cols", "Average tip", maximum=14)
    count = self.sheet("Pizza - City deliveries", rows=self.city, mark="Text",
      labels=(self.count_label,), city_sort=True, show_city=False, action=False)
    minutes = self.sheet("Pizza - City recorded minutes", rows=self.city, mark="Text",
      labels=(self.minutes,), tooltip=(self.count,), city_sort=True, show_city=False, action=False)
    hist = self.sheet("Pizza - Tip distribution", rows=self.count, cols=self.tip_bin,
      tooltip=(self.count, self.tip_bin, self.selected_median))
    self.axis(hist, self.count, "rows", "Deliveries")
    self.axis(hist, self.tip_bin, "cols", "Tip ($)", maximum=42)
    elem(hist.find("table"), "show-full-range").append(ET.Element("column"))
    hist.find("table/show-full-range/column").text = self.qualify("[Tip (bin)]")
    pane = hist.find("table/panes/pane")
    pane.insert(2, ET.Element("mark-sizing", {"custom-mark-size-in-axis-units": "1.0",
      "mark-alignment": "mark-alignment-left", "mark-sizing-setting": "marks-scaling-on",
      "use-custom-mark-size": "false"}))
    refline = ET.Element("reference-line", {"id": "pizza-median-tip", "axis-column": self.qualify(self.tip_bin),
      "value-column": self.qualify(self.selected_median), "scope": "per-table", "label-type": "custom",
      "label": "Median tip", "formula": "average", "z-order": "1", "enable-instant-analytics": "false"})
    pane.insert(list(pane).index(pane.find("style")), refline)
    trend = self.sheet("Pizza - Monthly average tip", rows=self.tip, cols=self.month,
      mark="Line", labels=(self.tip,), tooltip=(self.count,))
    self.axis(trend, self.tip, "rows", "Average tip", maximum=10)
    self.axis(trend, self.month, "cols", "", visible=False)
    volume = self.sheet("Pizza - Monthly deliveries", rows=self.count, cols=self.month,
      labels=(self.count,), color="#bbc7d2", label_size=9)
    self.axis(volume, self.count, "rows", "Deliveries")
    self.axis(volume, self.month, "cols", "Month")
    timing = self.sheet("Pizza - Delivery time distribution", rows=self.count, cols=self.time_bin,
      tooltip=(self.count, self.time_bin), color=BLUE)
    self.axis(timing, self.count, "rows", "Deliveries")
    self.axis(timing, self.time_bin, "cols", "Recorded order-to-delivery minutes", maximum=140)
    timing.find("table/panes/pane").insert(2, ET.Element("mark-sizing", {"custom-mark-size-in-axis-units": "1.0",
      "mark-alignment": "mark-alignment-left", "mark-sizing-setting": "marks-scaling-on",
      "use-custom-mark-size": "false"}))
    housing = self.sheet("Pizza - Housing average tip", rows=self.housing, cols=self.tip,
      labels=(self.tip,), tooltip=(self.count, self.minutes))
    self.axis(housing, self.tip, "cols", "Average tip", maximum=10)

  def zone_style(self, zone, background="#ffffff", border=False):
    style = elem(zone, "zone-style")
    for attr, value in (("background-color", background), ("border-color", MIST),
      ("border-style", "solid" if border else "none"), ("border-width", "1" if border else "0"),
      ("margin", "0"), ("padding", "6")):
      fmt(style, attr, value)

  def dashboard(self, name, variant):
    dashboard = ET.Element("dashboard", {"name": name, "enable-sort-zone-taborder": "true"})
    if name == MAIN:
      dashboard.append(copy.deepcopy(self.old_dashboard.find("repository-location")))
      simple = copy.deepcopy(self.old_dashboard.find("simple-id"))
    else:
      simple = ET.Element("simple-id", {"uuid": uid(name)})
    style_rule(elem(dashboard, "style"), "dashboard", background_color=BG)
    width, height = 1500, 1060
    elem(dashboard, "size", sizing_mode="fixed", minwidth=width, maxwidth=width, minheight=height, maxheight=height)
    zones = elem(dashboard, "zones")
    container = elem(zones, "zone", x=0, y=0, w=100000, h=100000, id=1, type_v2="layout-basic")
    zid = 1
    def zone(x,y,w,h,**attrs):
      nonlocal zid
      zid += 1
      return elem(container, "zone", x=round(x/width*100000), y=round(y/height*100000),
        w=round(w/width*100000), h=round(h/height*100000), id=zid, **attrs)
    def text(x,y,w,h,content,size=11,color=SLATE,bold=False,background=BG):
      z = zone(x,y,w,h,type_v2="text")
      ft = elem(z,"formatted-text")
      elem(ft,"run",content,fontname="Arial",fontsize=size,fontcolor=color,bold=str(bold).lower())
      self.zone_style(z,background)
      return z
    def sheet(x,y,w,h,name):
      z = zone(x,y,w,h,name=name,show_title="false",show_caption="false")
      self.zone_style(z)
      return z
    def panel(x,y,w,h):
      z=zone(x,y,w,h,type_v2="text")
      elem(z,"formatted-text")
      self.zone_style(z,border=True)
    text(22,12,780,24,"D A N I E L  S H O R T  /  D A T A  E X P L O R E R",9)
    text(22,36,980,48,"Pizza delivery" if variant == "overview" else "Pizza delivery - timing",27,NAVY,True)
    text(22,84,1100,28,"Tips, city comparisons, and delivery history",13)
    panel(16,122,1468,72)
    for x,w,field,mode in ((30,430,self.date,"range"),(480,235,self.city,"dropdown"),(735,235,self.housing,"dropdown")):
      z=zone(x,129,w,57,type_v2="filter",name="Pizza - Deliveries",param=self.qualify(field),
        mode=mode,show_all="true",values="database")
      self.zone_style(z)
    text(1000,143,460,34,"Reset: use Tableau Revert in the toolbar",10,BLUE,background="#ffffff")
    panel(16,205,1468,130)
    for x,w,title,sub,name in ((32,460,"Deliveries","Recorded orders","Pizza - Deliveries"),
      (532,450,"Median tip","Typical tip per delivery","Pizza - Median tip"),
      (1015,445,"Total tips","Across the selected period","Pizza - Total tips")):
      text(x,211,w,26,title,12,NAVY,True,"#ffffff")
      sheet(x,237,w,60,name)
      text(x,302,w,27,sub,11,SLATE,background="#ffffff")
    if variant == "overview":
      panel(16,346,826,385)
      text(30,350,790,30,"City comparison",16,NAVY,True,"#ffffff")
      text(30,382,790,24,"Average tip per delivery - sorted by delivery count",11,SLATE,background="#ffffff")
      text(35,414,130,24,"City",10,NAVY,True,"#ffffff")
      text(175,414,380,24,"Average tip",10,NAVY,True,"#ffffff")
      text(627,414,95,24,"Deliveries",10,NAVY,True,"#ffffff")
      text(725,414,110,24,"Recorded min",10,NAVY,True,"#ffffff")
      sheet(30,443,590,238,"Pizza - City average tip")
      sheet(627,443,90,238,"Pizza - City deliveries")
      sheet(727,443,98,238,"Pizza - City recorded minutes")
      text(30,690,795,32,"* Fewer than 30 deliveries; compare cautiously.  Select a city to filter other views.",10,SLATE,background="#ffffff")
      panel(852,346,632,385)
      text(866,350,598,30,"Tip distribution",16,NAVY,True,"#ffffff")
      text(866,382,598,24,"Delivery count - $2 bins - all tips included",11,SLATE,background="#ffffff")
      sheet(866,412,598,294,"Pizza - Tip distribution")
      panel(16,743,1468,275)
      text(30,748,1430,31,"Average tip over time",16,NAVY,True,"#ffffff")
      text(30,780,1430,25,"Monthly averages with delivery counts below",11,SLATE,background="#ffffff")
      sheet(30,808,1430,121,"Pizza - Monthly average tip")
      sheet(30,931,1430,75,"Pizza - Monthly deliveries")
    else:
      panel(16,346,930,465)
      text(30,352,900,33,"Recorded delivery time",16,NAVY,True,"#ffffff")
      text(30,386,900,28,"Order-to-delivery elapsed minutes - all observations retained",11,SLATE,background="#ffffff")
      sheet(30,420,900,375,"Pizza - Delivery time distribution")
      panel(958,346,526,465)
      text(972,352,498,34,"Housing comparison",16,NAVY,True,"#ffffff")
      text(972,389,498,27,"Average tip per delivery",11,SLATE,background="#ffffff")
      sheet(972,423,498,368,"Pizza - Housing average tip")
      text(30,838,1430,135,"About the data\nJuly 5, 2017 - September 23, 2018. Personal delivery history, not current market demand.\n34 recorded delivery times are zero minutes; one is 135 minutes. These records remain included.\nElapsed delivery time is not driver labor time. No hourly-earnings estimate is shown.",12)
    text(24,1024,1450,30,"Source: Pizza_Delivery workbook | Historical: Jul 2017-Sep 2018 | First/last months partial | Recorded time includes 34 zero-minute entries",9)
    self.zone_style(container,BG)
    dashboard.append(simple)
    return dashboard

  def run(self):
    self.build_sheets()
    worksheets = self.root.find("worksheets")
    worksheets.clear()
    worksheets.extend(self.sheets)
    dashboards = self.root.find("dashboards")
    dashboards.clear()
    dashboards.append(self.dashboard(MAIN,"overview"))
    dashboards.append(self.dashboard("Pizza Delivery Timing","timing"))
    actions = self.root.find("actions")
    actions.clear()
    a=elem(actions,"action",caption="Select a city",name=self.action)
    elem(a,"activation",auto_clear="true",type="on-select")
    elem(a,"source",dashboard=MAIN,type="sheet",worksheet="Pizza - City average tip")
    command=elem(a,"command",command="tsc:tsl-filter")
    elem(command,"param",name="special-fields",value="all")
    elem(command,"param",name="target",value=MAIN)
    windows=self.root.find("windows")
    windows.clear()
    windows.set("source-height","30")
    for s in self.sheets:
      window=elem(windows,"window",**{"class":"worksheet","name":s.get("name"),"hidden":"true"})
      cards=elem(window,"cards")
      edge=elem(cards,"edge",name="left")
      strip=elem(edge,"strip",size="160")
      for kind in ("pages","filters","marks"):
        elem(strip,"card",type=kind)
      elem(window,"viewpoint").append(ET.Element("zoom",{"type":"entire-view"}))
      elem(window,"simple-id",uuid=uid("window-"+s.get("name")))
    for d in dashboards:
      window=elem(windows,"window",**{"class":"dashboard","name":d.get("name"),"maximized":"true"})
      views=elem(window,"viewpoints")
      names=sorted({z.get("name") for z in d.findall(".//zone") if z.get("name") and not z.get("type-v2")})
      for name in names:
        elem(elem(views,"viewpoint",name=name),"zoom",type="entire-view")
      elem(window,"active",id="2")
      elem(window,"simple-id",uuid=uid("window-"+d.get("name")))
    assert self.original_connections == ET.tostring(self.source.find("connection"))
    assert self.original_objects == ET.tostring(self.source.find("object-graph"))
    return self.root


def check_schema(xml, path, baseline_xml):
  """Validate with the official XSD and explicit supplemental namespace imports.

  Tableau's published XSD imports user/XML namespaces without schema locations.
  Resolve XML attributes and allow user metadata without modifying its TWB rules.
  This is structural validation only; it does not validate calculation semantics.
  """
  from lxml import etree
  class Resolver(etree.Resolver):
    def resolve(self, url, public_id, context):
      if url.endswith("pizza-user.xsd"):
        return self.resolve_string('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://www.tableausoftware.com/xml/user"><xs:attributeGroup name="UserAttributes-AG"><xs:anyAttribute namespace="##any" processContents="lax"/></xs:attributeGroup></xs:schema>', context)
      if url.endswith("pizza-xml.xsd"):
        return self.resolve_string('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://www.w3.org/XML/1998/namespace"><xs:attribute name="base" type="xs:anyURI"/><xs:attribute name="lang" type="xs:language"/><xs:attribute name="space" type="xs:string"/><xs:attribute name="id" type="xs:ID"/></xs:schema>', context)
      return None
  parser = etree.XMLParser(no_network=True)
  parser.resolvers.add(Resolver())
  raw = path.read_text(encoding="utf-8")
  raw = raw.replace('<xs:import namespace="http://www.tableausoftware.com/xml/user"/>', '<xs:import namespace="http://www.tableausoftware.com/xml/user" schemaLocation="pizza-user.xsd"/>')
  raw = raw.replace('<xs:import namespace="http://www.w3.org/XML/1998/namespace"/>', '<xs:import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="pizza-xml.xsd"/>')
  schema = etree.XMLSchema(etree.fromstring(raw.encode(), parser))
  schema.validate(etree.fromstring(baseline_xml))
  baseline = {error.message for error in schema.error_log}
  valid = schema.validate(etree.fromstring(xml))
  errors = [error.message for error in schema.error_log]
  inherited = [error for error in errors if error in baseline]
  introduced = [error for error in errors if error not in baseline]
  return {"status": "passed" if valid else ("passed with inherited source deviation" if not introduced else "failed"),
    "schema": str(path), "errors": errors, "introduced_errors": introduced, "inherited_source_deviations": inherited,
    "namespace_resolution": "Supplemental XML and lax user metadata namespace declarations"}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input",type=Path,default=REPO/"design/tableau/sources/Pizza_Delivery.twbx")
  parser.add_argument("--output",type=Path,default=REPO/"design/tableau/pizza-delivery-revamped.twbx")
  parser.add_argument("--schema",type=Path,help="Optional official TWB XSD path (requires lxml).")
  args=parser.parse_args()
  if args.input.resolve() == args.output.resolve():
    raise ValueError("Input and output must be different files.")
  with zipfile.ZipFile(args.input) as archive:
    entries={name:archive.read(name) for name in archive.namelist()}
  twb_name=next(name for name in entries if name.lower().endswith(".twb"))
  baseline_xml=entries[twb_name]
  source_root=ET.fromstring(entries[twb_name])
  original_identity=copy.deepcopy(source_root.find("repository-location"))
  result=PizzaBuilder(source_root).run()
  ET.indent(result,space="  ")
  xml=ET.tostring(result,encoding="utf-8",xml_declaration=True)
  ET.fromstring(xml)
  schema_result = check_schema(xml,args.schema,baseline_xml) if args.schema else {"status":"not run","reason":"Pass --schema to validate with the official Tableau XSD."}
  if schema_result["status"] == "failed":
    raise ValueError("Workbook schema errors:\n" + "\n".join(schema_result["errors"]))
  assert ET.tostring(result.find("repository-location")).strip() == ET.tostring(original_identity).strip()
  entries[twb_name]=xml
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
    for name,data in entries.items():
      info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0))
      info.compress_type=zipfile.ZIP_DEFLATED
      archive.writestr(info,data)
  args.output.with_suffix(".twb").write_bytes(xml)
  with zipfile.ZipFile(args.input) as before, zipfile.ZipFile(args.output) as after:
    preserved={name:hashlib.sha256(after.read(name)).hexdigest() for name in before.namelist() if not name.endswith(".twb")}
    assert all(before.read(name)==after.read(name) for name in preserved)
    assert after.testzip() is None
  report={"input":str(args.input),"output":str(args.output),"source_url":"https://public.tableau.com/workbooks/Pizza_Delivery.twb?showVizHome=no",
    "package_integrity":"passed","xml_parse":"passed","extracts_unchanged":preserved,
    "source_connections_and_relationships":"unchanged","workbook_id":result.find("repository-location").get("id"),
    "dashboard_name":MAIN,"worksheets":[s.get("name") for s in result.findall("./worksheets/worksheet")],
    "schema_validation":schema_result,
    "native_render_validation":"pending","filter_interaction_validation":"pending",
    "limitations":["Requires Tableau Desktop Public Edition or a compatible native renderer before publishing.",
      "Rounded cards, exact concept typography, device layouts and navigation remain subject to native renderer QA.",
      "Reset uses Tableau's built-in Revert toolbar action; no inert custom reset control is shown."]}
  args.output.with_suffix(".validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
  print(json.dumps(report,indent=2))


if __name__ == "__main__":
  main()
