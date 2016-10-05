package com.statnlp.example.weak_semi_crf;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.util.ArrayList;

import com.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler.NodeType;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.ui.visualize.type.VNode;
import com.statnlp.ui.visualize.type.VisualizationViewerEngine;
import com.statnlp.ui.visualize.type.VisualizeGraph;



public class WeakSemiCRFViewer extends VisualizationViewerEngine {
	
	static double span_width = 100;

	static double span_height = 100;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	protected WeakSemiCRFInstance instance;
	
	protected String[] inputs;
	
	protected ArrayList<Span> outputs;
	
	public static String[] Aspects;

	public WeakSemiCRFViewer(NetworkCompiler compiler, FeatureManager fm,
			int TypeLength) {
		super(compiler, fm, TypeLength);

	}
	
	@SuppressWarnings("unchecked")
	protected void initData()
	{
		this.instance = (WeakSemiCRFInstance)super.instance;
		this.inputs = this.instance.getInputAsArray();
		this.outputs = (ArrayList<Span>)super.outputs;
		//WIDTH = instance.Length * span_width;
	}
	
	
	protected void initTypeColorMapping()
	{	
		colorMap[0] = Color.WHITE;
		colorMap[1] = Color.MAGENTA;
		colorMap[2] = Color.PINK;
//		colorMap[3] = Color.YELLOW;
//		colorMap[4] = Color.GREEN;
//		colorMap[5] = Color.LIGHT_GRAY;
//		colorMap[6] = Color.CYAN;
//		colorMap[7] = Color.WHITE;
//		colorMap[8] = Color.ORANGE;
		
	}

	@Override
	protected String label_mapping(int[] ids) {
//		int size = instance.size();
		int pos = ids[0]; // position
		int nodeId = ids[2];
		int nodeType = ids[1];
		if(nodeType == NodeType.LEAF.ordinal()){
			return "LEAF";
		} else if (nodeType == NodeType.ROOT.ordinal()){
			return "ROOT";
		}
//		ids[1]; // tag_id
//		ids[4]; // node type
		if(Label.get(nodeId).form.equals("O")){
			return inputs[pos];
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
//				int nodeId = ids[2];
				int nodeType = ids[1];
				if(nodeType == NodeType.LEAF.ordinal() || nodeType == NodeType.ROOT.ordinal()){
					node.color = colorMap[0];
				} else if(nodeType == NodeType.BEGIN.ordinal()){
					node.color = colorMap[1];
				} else if(nodeType == NodeType.END.ordinal()){
					node.color = colorMap[2];
				}
			}
		}
		
	}
	
	protected void initNodeCoordinate(VisualizeGraph vg)
	{
		for(VNode node : vg.getNodes())
		{
			int[] ids = node.ids;
//			int size = this.inputs.length;
			int pos = ids[0];
			int labelId = ids[2];
			int nodeType = ids[1];
			
			double x = pos * span_width * 2;
			if(nodeType == NodeType.END.ordinal()){
				x += 0.5*span_width;
			}
			int mappedId = labelId;
//			switch(mappedId){
//			case 0:
//				mappedId = 1; break;
//			case 1:
//				mappedId = 6; break;
//			case 6:
//				mappedId = 0; break;
//			}
			double y = mappedId * span_height + offset_height;
			if(nodeType == NodeType.ROOT.ordinal()){
				x = (pos + 1) * span_width * 2;
				y = 3 * span_height + offset_height;
			}
			
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
	}

}
