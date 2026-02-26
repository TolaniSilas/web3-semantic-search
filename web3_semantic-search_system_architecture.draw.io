<mxfile host="65bd71144e">
    <diagram id="-g8QbYmCl7cOoSTqkjCH" name="Page-1">
        <mxGraphModel dx="498" dy="785" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
            </root>
        </mxGraphModel>
    </diagram>
    <diagram id="web3-semantic-arch" name="Web3 Semantic Search">
        <mxGraphModel dx="1022" dy="594" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
                <!-- Client -->
                <mxCell id="client" value="Client / Web App" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
                    <mxGeometry x="80" y="360" width="140" height="50" as="geometry"/>
                </mxCell>
                <!-- FastAPI -->
                <mxCell id="api" value="FastAPI + Uvicorn&#10;(REST API)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
                    <mxGeometry x="340" y="350" width="160" height="70" as="geometry"/>
                </mxCell>
                <!-- ChromaDB -->
                <mxCell id="chroma" value="ChromaDB&#10;(vector store)" style="shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
                    <mxGeometry x="640" y="440" width="120" height="80" as="geometry"/>
                </mxCell>
                <!-- Groq -->
                <mxCell id="groq" value="Groq API&#10;(LLM + VLM)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;" vertex="1" parent="1">
                    <mxGeometry x="640" y="200" width="120" height="60" as="geometry"/>
                </mxCell>
                <!-- Pillow (small) -->
                <mxCell id="pillow" value="Pillow&#10;(image check)" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
                    <mxGeometry x="340" y="460" width="100" height="40" as="geometry"/>
                </mxCell>
                <!-- Arrows -->
                <mxCell id="e1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="client" target="api">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="e2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.75;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="api" target="chroma">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="e3" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.75;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="api" target="groq">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="e4" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;dashed=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="api" target="pillow">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <!-- Labels on edges -->
                <mxCell id="l1" value="analyze, index, search" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" vertex="1" connectable="0" parent="e1">
                    <mxGeometry x="-0.2" y="1" relative="1" as="geometry">
                        <mxPoint as="offset"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="l2" value="store / query" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" vertex="1" connectable="0" parent="e2">
                    <mxGeometry x="-0.2" y="1" relative="1" as="geometry">
                        <mxPoint as="offset"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="l3" value="LLM / VLM" style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];" vertex="1" connectable="0" parent="e3">
                    <mxGeometry x="-0.2" y="1" relative="1" as="geometry">
                        <mxPoint as="offset"/>
                    </mxGeometry>
                </mxCell>
                <!-- Title -->
                <mxCell id="title" value="Web3 Semantic Search — System Architecture" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=bold;fontSize=14;" vertex="1" parent="1">
                    <mxGeometry x="280" y="80" width="340" height="30" as="geometry"/>
                </mxCell>
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>