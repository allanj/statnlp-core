package com.statnlp.example.linear_crf;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.util.ArrayList;

import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.ui.visualize.type.VNode;
import com.statnlp.ui.visualize.type.VisualizationViewerEngine;
import com.statnlp.ui.visualize.type.VisualizeGraph;



public class LinearCRFViewer extends VisualizationViewerEngine {
	
	static double span_width = 100;

	static double span_height = 100;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	protected LinearCRFInstance instance;
	
	protected ArrayList<String[]> inputs;
	
	protected ArrayList<Label> outputs;
	
	public static String[] Aspects;

	public LinearCRFViewer(NetworkCompiler compiler, FeatureManager fm,
			int TypeLength) {
		super(compiler, fm, TypeLength);

	}
	
	protected void initData()
	{
		this.instance = (LinearCRFInstance)super.instance;
		this.inputs = (ArrayList<String[]>)super.inputs;
		this.outputs = (ArrayList<Label>)super.outputs;
		//WIDTH = instance.Length * span_width;
	}
	
	
	protected void initTypeColorMapping()
	{	
		colorMap[0] = Color.WHITE;
		colorMap[1] = Color.MAGENTA;
		colorMap[2] = Color.PINK;
		colorMap[3] = Color.YELLOW;
		colorMap[4] = Color.GREEN;
		colorMap[5] = Color.LIGHT_GRAY;
		colorMap[6] = Color.CYAN;
		colorMap[7] = Color.WHITE;
		colorMap[8] = Color.ORANGE;
		
	}

	@Override
	protected String label_mapping(int[] ids) {
		int size = instance.size();
		int pos = ids[0]-1; // position
		int nodeId = ids[1];
		int nodeType = ids[4];
		if(nodeType == NODE_TYPES.LEAF.ordinal()){
			return "LEAF";
		} else if (nodeType == NODE_TYPES.ROOT.ordinal()){
			return "ROOT";
		}
//		ids[1]; // tag_id
//		ids[4]; // node type
		if(Label.get(nodeId).getForm().equals("O")){
			return inputs.get(pos)[0];
		}
		return Label.get(nodeId).toString();
	}
	
	protected void initNodeColor(VisualizeGraph vg)
	{
		if (colorMap != null){
			for(VNode node : vg.getNodes())
			{
				int[] ids = node.ids;
//				int pos = ids[0];
				int nodeId = ids[1]-0;
				int nodeType = ids[4];
				if(nodeType != NODE_TYPES.NODE.ordinal()){
					node.color = colorMap[0];
				} else {
					node.color = colorMap[1 + (nodeId % 8)];
				}
			}
		}
		
	}
	
	protected void initNodeCoordinate(VisualizeGraph vg)
	{
		for(VNode node : vg.getNodes())
		{
			int[] ids = node.ids;
			int size = this.inputs.size();
			int pos = ids[0]-1;
			int labelId = ids[1];
			int nodeType = ids[4];
			
			double x = pos * span_width;
			int mappedId = labelId;
			switch(mappedId){
			case 0:
				mappedId = 1; break;
			case 1:
				mappedId = 6; break;
			case 6:
				mappedId = 0; break;
			}
			double y = mappedId * span_height + offset_height;
			if(nodeType == NODE_TYPES.ROOT.ordinal()){
				x = (pos + 1) * span_width;
				y = 3 * span_height + offset_height;
			}
			
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
	}

}
