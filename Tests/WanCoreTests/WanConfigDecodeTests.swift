import Foundation
import XCTest

@testable import WanCore

/// `sample_guide_scale` ships as a scalar (TI2V-5B, single expert) OR an array
/// (A14B, dual expert). `ScalarOrArrayDouble` must decode both into `[Double]`.
final class WanConfigDecodeTests: XCTestCase {
    private func decode(_ guideJSON: String) throws -> [Double] {
        let json = """
        {"model_type":"ti2v","model_version":"2.2","patch_size":[1,2,2],"text_len":512,
         "in_dim":48,"dim":3072,"ffn_dim":14336,"freq_dim":256,"text_dim":4096,"out_dim":48,
         "num_heads":24,"num_layers":30,"window_size":[-1,-1],"qk_norm":true,
         "cross_attn_norm":true,"eps":1e-6,"vae_stride":[4,16,16],"vae_z_dim":48,
         "dual_model":false,"boundary":0.0,"sample_shift":5.0,"sample_steps":40,
         "sample_guide_scale":\(guideJSON),"num_train_timesteps":1000,"sample_fps":24,
         "frame_num":81,"sample_neg_prompt":"x","max_area":901120,"t5_vocab_size":256384,
         "t5_dim":4096,"t5_dim_attn":4096,"t5_dim_ffn":10240,"t5_num_heads":64,
         "t5_num_layers":24,"t5_num_buckets":32}
        """
        return try JSONDecoder().decode(WanConfig.self, from: Data(json.utf8)).sampleGuideScale
    }

    func testScalarGuideScale() throws {
        XCTAssertEqual(try decode("5.0"), [5.0])  // TI2V-5B shape
    }

    func testArrayGuideScale() throws {
        XCTAssertEqual(try decode("[4.0, 3.0]"), [4.0, 3.0])  // A14B shape
    }
}
