/*
 * Ported from torch-rechub-scala: torchrec/models/matching/MAMBA.scala
 * (MAMBABlock — simplified selective SSM layer)
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MAMBABlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int dState;
    private final int dInner;
    private final int dtRank;
    private final LinearImpl xProj;
    private final LinearImpl ssmProj;
    private final LinearImpl dtProj;
    private final LinearImpl hProj;
    private final LinearImpl hProjOut;
    private final LinearImpl outProj;
    private final LayerNormImpl norm;
    private final DropoutImpl dropoutLayer;
    private final Tensor dParam;
    private final Tensor aLog;

    public MAMBABlock(int embedDim, int dState, float dropout, String device) {
        super("MAMBABlock");
        this.embedDim = embedDim;
        this.dState = dState;
        this.dInner = embedDim;
        this.dtRank = (int) Math.ceil(embedDim / 16.0);

        this.xProj = new LinearImpl(embedDim, dInner * 2L);
        xProj.to(new Device(device), false);
        register_module("xProj", xProj);

        this.ssmProj = new LinearImpl(dInner, dtRank + 2L * dState);
        ssmProj.to(new Device(device), false);
        register_module("ssmProj", ssmProj);

        this.dtProj = new LinearImpl(dtRank, dInner);
        dtProj.to(new Device(device), false);
        register_module("dtProj", dtProj);

        this.dParam = torch.ones(new long[]{dInner},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        register_parameter("dParam", dParam);

        this.hProj = new LinearImpl(dInner, dState);
        hProj.to(new Device(device), false);
        register_module("hProj", hProj);

        this.hProjOut = new LinearImpl(dState, dInner);
        hProjOut.to(new Device(device), false);
        register_module("hProjOut", hProjOut);

        this.outProj = new LinearImpl(dInner, embedDim);
        outProj.to(new Device(device), false);
        register_module("outProj", outProj);

        LongVector normShape = new LongVector(1);
        normShape.put(0, embedDim);
        this.norm = new LayerNormImpl(normShape);
        register_module("norm", norm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        // A matrix log-init: -log(d+1) for stability
        float[] vals = new float[dState * dState];
        for (int i = 0; i < dState * dState; i++) {
            int d = i % dState + 1;
            vals[i] = (float) (-Math.log(d + 1.0));
        }
        this.aLog = torch.tensor(vals,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .view(dState, dState);
        register_parameter("aLog", aLog);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            xProj.to(dev, false);
            ssmProj.to(dev, false);
            dtProj.to(dev, false);
            outProj.to(dev, false);
            norm.to(dev, false);
            dropoutLayer.to(dev, false);
            aLog.to(dev, ScalarType.Float);
            hProj.to(dev, false);
            hProjOut.to(dev, false);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, seq_len, embed_dim)
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);

        Tensor normed = norm.forward(x);
        Tensor xz = xProj.forward(normed);
        Tensor xHalf = xz.narrow(2, 0, dInner);
        Tensor zHalf = xz.narrow(2, dInner, dInner);
        Tensor gate = zHalf.sigmoid();

        Tensor ssmParams = ssmProj.forward(xHalf);
        Tensor dtPart = ssmParams.narrow(2, 0, dtRank);
        // bPart / cPart computed but not used in simplified recurrence (mirrors Scala)
        ssmParams.narrow(2, dtRank, dState);
        ssmParams.narrow(2, dtRank + dState, dState);

        Tensor dt = dtProj.forward(dtPart);
        torch.silu(dt); // dtAct — computed for API parity with Scala

        Tensor a = torch.exp(aLog); // (dState, dState)

        Tensor hInput = hProj.forward(xHalf); // (batch, seq_len, dState)

        Tensor aAvg = a.mean(0).unsqueeze(0); // (1, dState)
        Tensor aRep = aAvg.expand(batchSize, seqLen, dState);

        Tensor hInputT = hInput.transpose(0, 1); // (seq_len, batch, dState)
        Tensor h = hInputT.select(0, 0);
        List<Tensor> hiddenSeq = new ArrayList<>();
        hiddenSeq.add(h);

        for (int pos = 1; pos < seqLen; pos++) {
            Tensor xPos = hInputT.select(0, pos);
            Tensor aDecay = aRep.select(1, pos);
            h = h.mul(aDecay).add(xPos);
            hiddenSeq.add(h);
        }

        TensorVector stackedVec = new TensorVector();
        for (Tensor hs : hiddenSeq) {
            stackedVec.push_back(hs.unsqueeze(0));
        }
        Tensor hiddenStacked = torch.cat(stackedVec, 0);

        Tensor ssmOutPreProj = hiddenStacked.transpose(0, 1);
        Tensor ssmOut = hProjOut.forward(ssmOutPreProj);

        Tensor dParamView = dParam.view(1L, 1L, dInner);
        Tensor skipOut = xHalf.mul(dParamView);

        Tensor ssmGated = ssmOut.add(skipOut);
        Tensor gated = ssmGated.mul(gate);
        Tensor out = outProj.forward(gated);

        return dropoutLayer.forward(out);
    }
}
