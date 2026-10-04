# Requirements Document

## Introduction

NAB Sentry (by NAB AI) is an on-premises, CPU-only, fully offline natural-language search engine for video surveillance footage. It targets high-security facilities (banks, corporate headquarters, commercial yards) in markets with strict data-residency rules and limited cloud infrastructure. An operator types a free-text query such as "person in blue jacket entering reception" or "white pickup truck at gate" and receives ranked incident clips. Each clip shows the camera ID, absolute start and end timestamps, and an HTML5 player that seeks to the event start.

The MVP ingests local video files (RTSP live ingest comes later), runs on an Intel Core x86 laptop with 16 GB RAM and no GPU (Python 3.12.10), and must be demoable within 3 days. The processing pipeline is: video source → frame sampler → motion gate → object detector (YOLO11n ONNX) → CLIP embedder (OpenCLIP ViT-B/32) → SQLite metadata store plus FAISS vector index → search engine with temporal clustering → FastAPI service → single-page operator console.

Out of scope for the MVP: RTSP ring-buffer ingest, multi-object tracking (ByteTrack), VLM re-ranking, OpenVINO/INT8 optimisation, authentication and audit logging, and replacing the AGPL detector with an Apache-licensed one.

## Glossary

- **NAB_Sentry**: The whole system covered by this document.
- **Operator**: A security staff member who searches footage through the Console.
- **Integrator**: A technical user who installs, provisions, ingests footage into, evaluates, and demos NAB_Sentry.
- **Video_Source**: A component that implements the VideoSource protocol and yields decoded frames with their offsets. In the MVP the only implementation is File_Source.
- **File_Source**: The Video_Source implementation that reads local video files.
- **Sidecar_File**: A JSON file next to a video file, with the same base name, that contains `camera_id`, `label`, and `start_time` (ISO 8601).
- **Filename_Convention**: The fallback naming pattern `<CAMERA_ID>_<YYYYMMDDTHHMMSS>.<ext>` (example: `CAM01_20250101T080000.mp4`), which encodes the camera ID and the video start time.
- **Video_Start_Time**: The absolute wall-clock time of the first frame of a video, taken from the Sidecar_File or the Filename_Convention. A Video_Start_Time without a UTC offset is interpreted in the local time zone of the machine running the Ingest_Pipeline.
- **Frame_Offset**: The time of a frame in seconds from the start of its video.
- **Absolute_Timestamp**: Video_Start_Time plus Frame_Offset.
- **Ingest_Pipeline**: The component that turns a video file into stored metadata, thumbnails, playback files, and vectors.
- **Sampler**: The component that selects frames from a Video_Source at the Sample_Rate.
- **Sample_Rate**: The configured number of sampled frames per second of video (default 1, allowed range 0.1 to 30 inclusive).
- **Motion_Gate**: The component that decides whether a sampled frame has enough scene change to process further. It uses OpenCV MOG2 background subtraction or frame differencing on a downscaled grayscale copy of the frame.
- **Motion_Threshold**: The configured fraction of changed pixels at or above which the Motion_Gate passes a frame (default 0.02, allowed range 0.0 to 1.0 inclusive).
- **Keyframe_Interval**: The configured maximum gap in seconds of video time between passed frames (default 10 seconds, allowed range 1 to 600 seconds). The Motion_Gate passes a frame when no frame has passed within this interval, even without motion.
- **Detector**: The component that runs Ultralytics YOLO11n, exported to ONNX, on CPU.
- **Target_Classes**: The detector classes NAB_Sentry keeps: person, car, truck, bus, motorcycle, bicycle. Each one is a Target_Class.
- **Crop**: A rectangular sub-image of a frame bounded by a Detector bounding box of a Target_Class.
- **Embedder**: The component that computes OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) image and text embeddings.
- **Batch_Size**: The configured number of images the Embedder encodes per batch (integer from 1 to 64, default 8).
- **Embedding**: A 512-dimensional float32 vector with an L2 norm of 1 (tolerance 1e-4), produced by the Embedder.
- **Vector_Kind**: The type of image an Embedding was made from: `frame` (full frame) or `crop` (Crop).
- **Metadata_Store**: The SQLite database with the `cameras`, `videos`, `frames`, and `vectors` tables.
- **Vector_Index**: The FAISS `IndexIDMap2(IndexFlatIP(512))` index. Each vector ID equals a row ID in the `vectors` table.
- **Search_Filter**: An optional restriction on a search by camera ID, Absolute_Timestamp range, and/or Target_Class.
- **Allowed_ID_Set**: The set of vector IDs in the Metadata_Store whose frame and detection metadata satisfy a Search_Filter.
- **Search_Engine**: The component that encodes a query, searches the Vector_Index, and clusters the hits into Events.
- **Prompt_Templates**: The text templates that wrap a query before text encoding: `"{q}"` and `"a CCTV photo of {q}"`.
- **Hit**: One vector match returned by the Vector_Index, made of a vector ID and a cosine similarity.
- **Top_K**: The configured number of Hits fetched per search (default 300).
- **Event**: A group of Hits from the same camera and video whose Frame_Offsets are linked by gaps of no more than the Merge_Gap.
- **Merge_Gap**: The configured maximum gap between consecutive Hits in one Event (default 8 seconds).
- **Event_Padding**: The configured time added before and after an Event (default 3 seconds), clamped to the video bounds.
- **Event_Score**: The mean of the highest three Hit similarities in an Event, or the mean of all its Hits if it has fewer than three.
- **Relative_Score**: Event_Score minus the median similarity of all Top_K Hits for the query.
- **Label_Boost**: The configured value added to an Event_Score when a query word matches a word of the Event's camera label (case-insensitive).
- **API_Server**: The FastAPI application that exposes the search and media endpoints.
- **Console**: The single-page operator interface (plain HTML, JavaScript, CSS, no build step) served by the API_Server.
- **Thumbnail**: A JPEG image 320 pixels wide, kept at the source aspect ratio, stored under `data/thumbs/`.
- **Playback_File**: A browser-playable H.264 MP4 with the moov atom at the start (faststart), made with imageio-ffmpeg and stored under `data/playback/`.
- **Src_Hash**: A content hash of a source video file, used to detect files that are already ingested.
- **Model_Fetcher**: The script `scripts/fetch_models.py`, which downloads model weights into `models/` once, on a connected machine.
- **Query_Log**: A local append-only file recording each search request. It is the basis for a post-MVP audit log.
- **Config**: The module `config.py`, which holds all tunable parameters.
- **Synthetic_Generator**: The script that uses OpenCV to generate synthetic test videos with a matching `ground_truth.json`.
- **Evaluator**: The script `scripts/evaluate.py`, which runs the queries in `eval/queries.yaml` and reports quality and latency metrics.
- **Benchmark_Tool**: The script `scripts/benchmark.py`, which measures per-stage throughput on the target hardware.
- **Target_Hardware**: An Intel Core x86 laptop with 16 GB RAM, no GPU, running Windows and Python 3.12.10.

## Requirements

### Requirement 1: Video Source and Camera Metadata Resolution

**User Story:** As an Integrator, I want each video file to carry its camera ID, label, and absolute start time, so that search results show the correct camera and wall-clock time.

#### Acceptance Criteria

1. THE File_Source SHALL implement the VideoSource protocol, yielding each decoded frame in decode order together with its Frame_Offset in seconds, where the first yielded frame has a Frame_Offset of 0.0 (tolerance 0.001 seconds) and each later Frame_Offset is greater than or equal to the one before it.
2. WHEN a video file has a valid Sidecar_File, THE Ingest_Pipeline SHALL take the camera ID, camera label, and Video_Start_Time from the Sidecar_File. A valid Sidecar_File is in the same directory as the video, has the same base name with a `.json` extension, parses as a single JSON object, and holds a `camera_id` of 1 to 64 characters made only of letters, digits, and hyphens, a `label` of 1 to 128 characters that is not only whitespace, and a `start_time` that is a valid ISO 8601 date-time.
3. WHEN a video file has no valid Sidecar_File and its name matches the Filename_Convention, THE Ingest_Pipeline SHALL take the camera ID and Video_Start_Time from the file name and use the camera ID as the camera label. A name matches the Filename_Convention only if it has a camera ID of 1 to 64 letters, digits, and hyphens, then one underscore, then a `YYYYMMDDTHHMMSS` timestamp that is a real calendar date and time (for example, month 01 to 12 and hour 00 to 23), then a file extension.
4. WHEN a video file has both a valid Sidecar_File and a name that matches the Filename_Convention, THE Ingest_Pipeline SHALL use the Sidecar_File values and ignore the values encoded in the file name.
5. IF a video file has no valid Sidecar_File and its name does not match the Filename_Convention, THEN THE Ingest_Pipeline SHALL skip the file, log the file path with the reason "unresolved camera metadata", write no rows, Thumbnails, Playback_Files, or vectors for that file, and go on to ingest the remaining files.
6. IF a Sidecar_File exists but does not parse as a JSON object, is missing a required field, or holds a field value that breaks the rules in criterion 2, THEN THE Ingest_Pipeline SHALL log the file path and the name of each invalid field (or that the file is unparseable), and SHALL treat the video as having no valid Sidecar_File, so that criterion 3 or criterion 5 applies.
7. FOR ALL camera IDs that are valid under criterion 3 and all start times with whole-second precision in years 0001 to 9999, formatting a file name and then parsing it SHALL return the original camera ID and start time (round-trip property).
8. FOR ALL valid Sidecar_File objects, serialising to JSON and then parsing SHALL return an equivalent object, meaning identical `camera_id` and `label` strings and a `start_time` with the same date, time, and UTC offset (or the same lack of one) (round-trip property).
9. WHEN a camera ID is resolved that has no row in the `cameras` table, THE Ingest_Pipeline SHALL create exactly one row in the `cameras` table with that camera ID and label. When the camera ID already has a row, it SHALL create no new row.
10. IF a resolved camera ID already exists in the `cameras` table with a different label, THEN THE Ingest_Pipeline SHALL leave the stored row unchanged, log the camera ID, the stored label, and the new label, and continue ingesting the video under the existing camera row.
11. IF the File_Source cannot open a video file or decodes zero frames from it, THEN THE Ingest_Pipeline SHALL skip the file, log the file path with the reason "unreadable video", write no rows, Thumbnails, Playback_Files, or vectors for that file, and go on to ingest the remaining files.
12. WHEN a Video_Start_Time is resolved without a UTC offset (always the case for the Filename_Convention), THE Ingest_Pipeline SHALL interpret it in the local time zone of the machine running the Ingest_Pipeline when computing Absolute_Timestamps.

### Requirement 2: Frame Sampling

**User Story:** As an Integrator, I want frames sampled at a configurable rate, so that I can trade ingest speed against how fine-grained the timeline is.

#### Acceptance Criteria

1. WHEN the Sampler processes a video, THE Sampler SHALL select, for each target time k ÷ Sample_Rate seconds (k = 0, 1, 2, … while the target time is no greater than the video duration), the first decoded frame whose Frame_Offset is at or after that target time and that has not already been selected, using the Sample_Rate set in the Config (default 1 frame per second; allowed range 0.1 to 30 frames per second inclusive).
2. WHEN the Sampler selects a frame, THE Sampler SHALL record a Frame_Offset equal to the zero-based frame index divided by the source frame rate, in seconds.
3. FOR ALL selected frames, THE Ingest_Pipeline SHALL store an Absolute_Timestamp that differs from Video_Start_Time plus Frame_Offset by no more than 1 millisecond (property).
4. FOR ALL selected frames of a video, the Frame_Offsets SHALL increase strictly in selection order and each SHALL fall between 0 and the video duration inclusive (invariant).
5. FOR ALL fully decodable videos with duration D seconds, source frame rate at or above Sample_Rate r, the Sampler SHALL select between floor(D × r) and floor(D × r) + 1 frames (property).
6. IF a video's source frame rate is lower than the Sample_Rate, THEN THE Sampler SHALL select every decoded frame exactly once.
7. IF a frame cannot be decoded, THEN THE File_Source SHALL log the video path and the Frame_Offset of the failed frame, store no record for that frame, and continue with the next frame, so that the selected-frame count for that video falls below the lower bound in criterion 5 by at most the number of failed frames.
8. IF the Sample_Rate in the Config is non-numeric, outside 0.1 to 30 frames per second, or missing, THEN THE Ingest_Pipeline SHALL refuse to start ingest, report an error indicating the invalid Sample_Rate value, and store no frames.
9. IF a video's source frame rate is missing, zero, or negative in its container metadata, THEN THE Ingest_Pipeline SHALL log the video path with an error indicating the unknown frame rate, skip that video without storing any of its frames, and continue with the next video.
10. IF a video file cannot be opened or yields zero decodable frames, THEN THE Ingest_Pipeline SHALL log the video path with an error indicating the video is unreadable, store no frames for that video, and continue with the next video.

### Requirement 3: Motion Gating

**User Story:** As an Integrator, I want static frames skipped before the expensive models run, so that ingest fits the CPU budget.

#### Acceptance Criteria

1. WHEN the Sampler selects a frame, THE Motion_Gate SHALL compute a changed-pixel fraction between 0.0 and 1.0 inclusive on a grayscale copy of the frame downscaled to the configured gate width (default 320 pixels, allowed range 64 to 1920 pixels) with the source aspect ratio kept, comparing it only against earlier sampled frames of the same video, and without upscaling frames whose source width is at or below the gate width.
2. WHEN the Motion_Gate computes a changed-pixel fraction at or above the Motion_Threshold (default 0.02, allowed range 0.0 to 1.0 inclusive), THE Motion_Gate SHALL pass the frame to the Detector and the Embedder.
3. WHEN the Sampler selects a frame whose Frame_Offset is at least Keyframe_Interval seconds (default 10 seconds, allowed range 1 to 600 seconds) after the Frame_Offset of the last frame of the same video that passed the Motion_Gate, THE Motion_Gate SHALL pass that frame to the Detector and the Embedder as a keyframe, measuring the interval in video time (Frame_Offset) and not in wall-clock processing time.
4. WHEN the Sampler selects the first frame of a video, THE Motion_Gate SHALL pass that frame to the Detector and the Embedder regardless of the Motion_Threshold, and SHALL reset its motion and keyframe state so that no state carries over from a previously ingested video.
5. FOR ALL sequences of identical sampled frames at the start of a video spanning D seconds of Frame_Offset, where Keyframe_Interval is an integer multiple of the sample period (1 / Sample_Rate), THE Motion_Gate SHALL pass exactly 1 + floor(D / Keyframe_Interval) frames: the first frame plus one keyframe per elapsed Keyframe_Interval (property).
6. WHEN the Ingest_Pipeline finishes ingesting a video, THE Ingest_Pipeline SHALL record in the Metadata_Store `videos` record for that video the number of sampled frames and the number of frames passed by the Motion_Gate, where the passed count is at least 1 and at most the sampled count (invariant) for any video with at least one sampled frame.
7. IF a sampled frame has a changed-pixel fraction below the Motion_Threshold and is neither the first frame of its video nor due as a keyframe, THEN THE Motion_Gate SHALL discard the frame so that it is not passed to the Detector or the Embedder, and THE Ingest_Pipeline SHALL store no `frames` row, no Thumbnail, and no Embedding for it.
8. IF the configured Motion_Threshold, Keyframe_Interval, or gate width is outside its allowed range when ingest starts, THEN THE Ingest_Pipeline SHALL refuse to ingest, report an error naming the out-of-range parameter, and leave the Metadata_Store and Vector_Index unchanged.
9. IF the Sampler yields an empty or zero-size frame, THEN THE Motion_Gate SHALL discard that frame without passing it to the Detector or the Embedder, count it as sampled but not passed, keep its motion and keyframe state from the last valid frame, and continue with the next sampled frame of the same video.

### Requirement 4: Object Detection

**User Story:** As an Operator, I want people and vehicles detected and cropped, so that queries about a specific person or vehicle match that object rather than the whole scene.

#### Acceptance Criteria

1. WHEN the Motion_Gate passes a frame, THE Detector SHALL run YOLO11n inference through ONNX Runtime on CPU on that frame and return a list of zero or more detections, each made of a class name, a confidence between 0.0 and 1.0, and a bounding box in pixel coordinates of the original frame (not the resized model input).
2. THE Detector SHALL keep only detections whose class is one of the Target_Classes and whose confidence is at or above the configured detection threshold (default 0.35, valid range 0.0 to 1.0 inclusive), and SHALL discard all other detections.
3. FOR ALL kept detections, THE Detector SHALL return an integer-pixel bounding box clamped to the frame bounds (0 ≤ x1 < x2 ≤ frame width, 0 ≤ y1 < y2 ≤ frame height), with a width and height of at least 1 pixel, and SHALL discard any detection whose clamped box would be narrower or shorter than 1 pixel (invariant).
4. WHEN a frame has more kept detections than the configured per-frame maximum (default 10, valid range 1 to 100), THE Detector SHALL keep only the detections with the highest confidence, so that the number kept equals that maximum, and SHALL discard the rest.
5. WHEN the Detector keeps a detection, THE Ingest_Pipeline SHALL store its class name, confidence, and bounding box in the Metadata_Store, linked to the frame it came from, so that the stored count of detections for that frame equals the number of kept detections.
6. IF the ONNX model file is missing from `models/` when the Detector is initialised, THEN THE Detector SHALL raise an error naming the expected path and the Model_Fetcher command before any frame of the video is processed, and THE Ingest_Pipeline SHALL write no frame, detection, or vector records for that video.
7. WHEN the Detector returns no kept detections for a passed frame, THE Ingest_Pipeline SHALL store the frame in the Metadata_Store with zero linked detections and SHALL continue processing the frame as a full frame.
8. IF ONNX Runtime inference raises an error on a passed frame, THEN THE Ingest_Pipeline SHALL record a warning identifying the video and Frame_Offset, store the frame with zero linked detections, and continue with the next sampled frame without stopping the ingest of the video.
9. IF the configured detection threshold is outside 0.0 to 1.0 or the configured per-frame maximum is outside 1 to 100, THEN THE Detector SHALL raise an error at initialisation naming the invalid parameter and its valid range, and SHALL process no frames.

### Requirement 5: Image and Text Embedding

**User Story:** As an Operator, I want frames, crops, and my query mapped into one vector space, so that a text query can be matched against visual content.

#### Acceptance Criteria

1. WHEN the Motion_Gate passes a frame, THE Embedder SHALL compute exactly one Embedding of Vector_Kind `frame` from the full, uncropped frame.
2. WHEN the Detector keeps a detection whose Crop, after clamping its bounding box to the frame bounds, is at least 1 pixel wide and at least 1 pixel high, THE Embedder SHALL compute exactly one Embedding of Vector_Kind `crop` from that Crop.
3. THE Embedder SHALL return, for every image and every query string it encodes, a 512-dimensional float32 vector whose 512 components are all finite (no NaN or infinity) and whose L2 norm is within 1e-4 of 1 (invariant).
4. WHEN a query is encoded, THE Embedder SHALL encode the query once with each of the two Prompt_Templates, truncating each templated string to the text encoder's 77-token context length, take the unweighted element-wise mean of the two resulting vectors, and L2-normalise that mean so the result satisfies criterion 3.
5. THE Embedder SHALL process images in batches of the configured batch size (an integer from 1 to 64, default 8), where every batch except the last holds exactly the configured batch size and the last batch holds the remaining 1 to batch-size images.
6. THE Embedder SHALL produce, for every image, a vector from single-image encoding and a vector from encoding the same image inside a batch whose cosine similarity is at least 0.999 (property).
7. THE Embedder SHALL load the OpenCLIP ViT-B/32 `laion2b_s34b_b79k` weights only from the local `models/` directory and SHALL make no network request when loading weights or computing Embeddings.
8. IF the OpenCLIP ViT-B/32 `laion2b_s34b_b79k` weights are absent from or unreadable in the `models/` directory when the Embedder loads, THEN THE Embedder SHALL stop loading without attempting a download, produce no Embeddings, and raise an error indicating the missing weights and that the Model_Fetcher must be run.
9. IF a query string is empty or contains only whitespace after trimming, THEN THE Embedder SHALL reject the query with an error indicating an empty query and SHALL return no vector.
10. IF a kept detection's Crop, after clamping its bounding box to the frame bounds, is less than 1 pixel wide or less than 1 pixel high, THEN THE Embedder SHALL skip that Crop, compute no `crop` Embedding for it, and continue embedding the remaining frames and Crops.

### Requirement 6: Metadata Storage, Thumbnails, and Playback Files

**User Story:** As an Integrator, I want every indexed frame traceable to its video, camera, time, thumbnail, and playable file, so that results can be shown and played back.

#### Acceptance Criteria

1. THE Metadata_Store SHALL hold the tables `cameras`, `videos`, `frames`, and `vectors`, with enforced foreign keys from `videos` to `cameras`, from `frames` to `videos`, and from `vectors` to `frames`, such that an insert of a row referencing a non-existent parent row is rejected.
2. WHEN a frame passes the Motion_Gate, THE Ingest_Pipeline SHALL store exactly one `frames` row holding the video ID, the Frame_Offset in seconds with at least millisecond precision, the Absolute_Timestamp in ISO 8601 equal to Video_Start_Time plus Frame_Offset (tolerance 1 millisecond), and the Thumbnail path, and SHALL store no `frames` row for a sampled frame that the Motion_Gate rejects.
3. WHEN an Embedding is stored, THE Ingest_Pipeline SHALL store exactly one `vectors` row holding the frame ID and Vector_Kind, and, for Vector_Kind `crop`, the Target_Class of the detection and the bounding box as pixel coordinates in the source frame that lie within the frame width and height.
4. WHEN a frame passes the Motion_Gate, THE Ingest_Pipeline SHALL write one Thumbnail of that frame to `data/thumbs/`, with a file name unique to that `frames` row, 320 pixels wide, and with a height within 1 pixel of the source aspect ratio.
5. WHEN a video is ingested, THE Ingest_Pipeline SHALL write exactly one Playback_File for the video to `data/playback/` using imageio-ffmpeg, encoded as H.264 with faststart so that the moov atom precedes the media data, and SHALL store the Playback_File path in that video's `videos` row.
6. FOR ALL Playback_Files, the duration SHALL be within 0.5 seconds of the source video duration (property).
7. FOR ALL states of NAB_Sentry after an ingest completes or is rolled back, including after the Metadata_Store and Vector_Index are reloaded on restart, FOR ALL rows in the `vectors` table the Vector_Index SHALL hold exactly one vector with the same ID, and FOR ALL vectors in the Vector_Index the `vectors` table SHALL hold exactly one row with the same ID (invariant).
8. IF ingest of a video fails before completion, including a decode failure, a Thumbnail write failure, or a Playback_File encode failure, THEN THE Ingest_Pipeline SHALL remove that video's `videos`, `frames`, and `vectors` rows, its Vector_Index entries, its Thumbnails, and its Playback_File, leave the rows, vectors, and files of all other videos unchanged, and log the failure with the video path and the failure reason.
9. WHEN ingest of a video starts, THE Ingest_Pipeline SHALL store one `cameras` row for the video's camera ID if no such row exists, holding the camera ID and label, and one `videos` row holding the camera ID, source file path, Src_Hash, Video_Start_Time, and source duration in seconds.
10. FOR ALL `frames` rows, the stored Thumbnail path SHALL refer to an existing JPEG file, and FOR ALL `videos` rows of completed ingests, the stored Playback_File path SHALL refer to an existing MP4 file (invariant).

### Requirement 7: Idempotent Re-Ingest

**User Story:** As an Integrator, I want to re-run ingest on the same folder safely, so that I can add footage without creating duplicates.

#### Acceptance Criteria

1. WHEN the Ingest_Pipeline begins processing a video file, THE Ingest_Pipeline SHALL compute the file's Src_Hash before writing any `videos`, `frames`, or `vectors` row, Thumbnail, Playback_File, or Vector_Index entry for that file, and SHALL store the Src_Hash in the file's `videos` row.
2. THE Ingest_Pipeline SHALL produce the same Src_Hash for two video files with identical byte content regardless of their file paths or file names, and different Src_Hashes for two video files whose byte content differs.
3. IF a video file's Src_Hash already exists in the `videos` table, or matches a file already ingested earlier in the same ingest run, THEN THE Ingest_Pipeline SHALL skip the file without writing any row, Thumbnail, Playback_File, or Vector_Index entry, SHALL log a message containing "already indexed" and the file path, and SHALL continue with the next file.
4. FOR ALL sets of 1 or more video files, ingesting the set twice SHALL leave the row counts of the `cameras`, `videos`, `frames`, and `vectors` tables, the Vector_Index size, and the number of files under `data/thumbs/` and `data/playback/` equal to their values after the first ingest (idempotence property).
5. FOR ALL sets of 1 or more video files, ingesting them in any order SHALL produce the same set of (camera ID, Absolute_Timestamp at millisecond precision, Vector_Kind) records, with row IDs and vector IDs allowed to differ (confluence property).
6. WHEN an ingest run processes a folder that contains both previously ingested video files and video files whose Src_Hash is absent from the `videos` table, THE Ingest_Pipeline SHALL ingest only the files whose Src_Hash is absent, and the resulting Metadata_Store row counts and Vector_Index size SHALL equal those produced by a single ingest of the full folder into an empty Metadata_Store and Vector_Index.
7. IF ingest of a video fails after processing of that video has started, THEN THE Ingest_Pipeline SHALL leave no `videos`, `frames`, or `vectors` row, Thumbnail, Playback_File, or Vector_Index entry for that video, SHALL log an error message containing the file path and the failure reason, and SHALL continue with the next file, so that a later ingest run processes the video again instead of skipping it.
8. IF a video file cannot be read to compute its Src_Hash, THEN THE Ingest_Pipeline SHALL log an error message containing the file path, SHALL skip the file without writing any row, Thumbnail, Playback_File, or Vector_Index entry, and SHALL continue with the next file.

### Requirement 8: Vector Index and Filtered Search

**User Story:** As an Operator, I want to restrict searches by camera, time range, and object class, so that I only see results relevant to the incident I am investigating.

#### Acceptance Criteria

1. THE Vector_Index SHALL be a FAISS `IndexIDMap2(IndexFlatIP(512))` whose vector IDs equal the `vectors` table row IDs.
2. WHEN a search has no Search_Filter, THE Search_Engine SHALL return the Hits for the min(Top_K, number of indexed vectors) indexed vectors with the highest inner product against the query vector, ordered by descending similarity.
3. WHEN a search has a Search_Filter, THE Search_Engine SHALL compute the Allowed_ID_Set from the Metadata_Store as the vectors that satisfy every specified filter part (camera, time range, class) combined with logical AND, search the Vector_Index restricted to that set through an `IDSelectorBatch`, and return at most Top_K Hits ordered by descending similarity.
4. IF restricted search through `IDSelectorBatch` is unavailable or raises an error, THEN THE Search_Engine SHALL search unrestricted with a fallback K equal to min(number of indexed vectors, 10 × Top_K), drop Hits outside the Allowed_ID_Set, and return the first Top_K remaining Hits ordered by descending similarity, without returning an error to the caller.
5. FOR ALL Search_Filters and query vectors, every returned Hit ID SHALL be a member of the Allowed_ID_Set (property).
6. FOR ALL Search_Filters and query vectors, the restricted search and the post-filter fallback SHALL return the same Hit IDs in the same order, with similarities equal within 1e-5, when the fallback K is at least the index size, disregarding order among Hits whose similarities differ by less than 1e-6 (model-based property).
7. FOR ALL query vectors, the Hits returned by the Vector_Index SHALL equal a brute-force inner-product ranking over the stored Embeddings in IDs and order, with similarities equal within 1e-5, disregarding order among Hits whose similarities differ by less than 1e-6 (model-based property).
8. WHEN the Search_Filter has a class filter, THE Search_Engine SHALL include only vectors of Vector_Kind `crop` whose detection class equals the filter class, and SHALL exclude all vectors of Vector_Kind `frame`.
9. WHEN the Search_Filter has a time range, THE Search_Engine SHALL include only vectors whose frame Absolute_Timestamp is greater than or equal to the range start (if given) and less than or equal to the range end (if given), treating an omitted start or end as unbounded on that side.
10. WHEN ingest of a video finishes successfully, THE Ingest_Pipeline SHALL save the Vector_Index to disk such that the saved index size equals the number of rows in the `vectors` table.
11. FOR ALL Vector_Index states, saving to disk and then loading SHALL give an index with the same size, the same IDs, and identical Hit IDs and similarities for any query vector (round-trip property).
12. WHEN the Search_Filter yields an empty Allowed_ID_Set, THE Search_Engine SHALL return an empty list of Hits with no error indication and without querying the Vector_Index.
13. WHEN the Search_Filter has a camera filter, THE Search_Engine SHALL include only vectors whose frame belongs to a video of the specified camera ID, and a camera ID absent from the `cameras` table SHALL yield an empty Allowed_ID_Set.
14. IF a Search_Filter has a time range whose start is later than its end, or a class filter that is not one of the Target_Classes, THEN THE Search_Engine SHALL reject the search with an error indicating which filter part is invalid, without querying the Vector_Index.
15. IF the Vector_Index file is missing, cannot be loaded, or its set of vector IDs differs from the set of row IDs in the `vectors` table, THEN THE Search_Engine SHALL refuse to execute searches and report an error indicating that the index is unavailable or inconsistent with the Metadata_Store.

### Requirement 9: Temporal Clustering and Ranking

**User Story:** As an Operator, I want neighbouring matching frames merged into one incident clip with a score, so that I review incidents rather than hundreds of near-duplicate frames.

#### Acceptance Criteria

1. WHEN the Search_Engine receives a list of 1 to Top_K Hits for a query, THE Search_Engine SHALL group the Hits by (camera ID, video ID) and sort each group by Frame_Offset ascending. Hits with equal Frame_Offsets (for example, a `frame` and a `crop` Vector_Kind from the same frame) stay in the same group.
2. WHEN two consecutive Hits in a sorted group have Frame_Offsets that differ by less than or equal to the Merge_Gap, THE Search_Engine SHALL place both Hits in the same Event. A difference of exactly the Merge_Gap counts as merged.
3. WHEN two consecutive Hits in a sorted group have Frame_Offsets that differ by more than the Merge_Gap, THE Search_Engine SHALL end the current Event at the earlier Hit and start a new Event at the later Hit.
4. THE Search_Engine SHALL produce only Events in which every Hit has the Event's camera ID and video ID (invariant).
5. THE Search_Engine SHALL produce Events such that, for every pair of Events from the same video, the Frame_Offset gap between the last Hit of the earlier Event and the first Hit of the later Event is greater than the Merge_Gap (invariant).
6. THE Search_Engine SHALL assign every input Hit to exactly one Event, so that the total number of Hits across all Events equals the number of input Hits (partition invariant).
7. WHEN an Event is formed, THE Search_Engine SHALL set the Event start to the earliest Hit Frame_Offset minus Event_Padding, clamped to at least 0 seconds, and the Event end to the latest Hit Frame_Offset plus Event_Padding, clamped to at most the video duration in seconds.
8. THE Search_Engine SHALL produce Events that satisfy 0 ≤ event start ≤ earliest Hit Frame_Offset ≤ latest Hit Frame_Offset ≤ event end ≤ video duration (invariant). This includes single-Hit Events, where the earliest and latest Hit Frame_Offsets are equal.
9. WHEN an Event is formed, THE Search_Engine SHALL compute its Event_Score from the Hit similarities of that Event, and its Relative_Score as the final Event_Score (after any Label_Boost) minus the median similarity of all Hits received for the query. "All Hits received" means up to Top_K, or fewer if the Vector_Index or the Allowed_ID_Set returned fewer.
10. IF at least one query word of 3 or more characters equals a word of the Event's camera label, ignoring case and with words split on whitespace and punctuation, THEN THE Search_Engine SHALL add the Label_Boost to that Event's Event_Score once. The boost is added once however many words match.
11. IF no query word of 3 or more characters equals a word of the Event's camera label, THEN THE Search_Engine SHALL leave the Event_Score unchanged.
12. WHEN clustering completes, THE Search_Engine SHALL return Events ordered by final Event_Score descending. Ties are broken first by earlier Absolute_Timestamp of the Event start, then by camera ID ascending.
13. THE Search_Engine SHALL produce the same set of Events from the same Hits in any input order: identical camera ID, video ID, member Hits, start, end, and ordering, with Event_Score and Relative_Score values equal within 1e-6 (confluence property).
14. WHEN an Event is formed, THE Search_Engine SHALL use as its representative Thumbnail the Thumbnail of the frame of the Hit with the highest similarity in that Event. If several Hits share the highest similarity, the Hit with the earliest Frame_Offset is used.
15. IF the Search_Engine receives an empty Hit list (for example, because the Vector_Index is empty or the Allowed_ID_Set is empty), THEN THE Search_Engine SHALL return an empty list of Events without error.

### Requirement 10: Search API

**User Story:** As an Operator, I want a local HTTP API for search, so that the Console can query footage and show ranked incidents.

#### Acceptance Criteria

1. THE API_Server SHALL bind only to the address 127.0.0.1, on the port set in Config, and SHALL refuse connections made to any other network interface of the host.
2. WHEN a client sends `GET /api/health`, THE API_Server SHALL respond with HTTP 200 and a JSON body that reports the number of indexed videos as a non-negative integer, the number of vectors in the Vector_Index as a non-negative integer, and whether the Detector and Embedder models are loaded as a boolean.
3. WHEN a client sends `GET /api/cameras`, THE API_Server SHALL respond with HTTP 200 and a JSON list that contains each camera in the Metadata_Store exactly once with its camera ID and label, or an empty list if the Metadata_Store contains no cameras.
4. WHEN a client sends `GET /api/search` with parameter `q` and the optional parameters `camera`, `start`, `end`, `cls`, and `limit`, THE API_Server SHALL respond with HTTP 200 and a JSON list of at most `limit` Events (default 20), ordered by Event_Score plus any Label_Boost in descending order.
5. THE API_Server SHALL return for each Event the camera ID, camera label, video ID, event start and end as Frame_Offsets in seconds, event start and end as ISO 8601 Absolute_Timestamps, Event_Score, Relative_Score, Thumbnail URL, and Playback_File URL.
6. FOR ALL returned Events, the ISO 8601 start SHALL equal Video_Start_Time plus the event start offset within 1 millisecond, the ISO 8601 end SHALL equal Video_Start_Time plus the event end offset within 1 millisecond, and the event start offset SHALL be less than or equal to the event end offset (property).
7. IF `q` is missing, empty, contains only whitespace, or is longer than 256 characters, THEN THE API_Server SHALL respond with HTTP 422 and a JSON error message identifying `q` as the invalid parameter, and SHALL not search the Vector_Index.
8. IF `start` or `end` is not valid ISO 8601, or `start` is later than `end`, THEN THE API_Server SHALL respond with HTTP 422 and a JSON error message identifying the invalid parameter, and SHALL not search the Vector_Index.
9. IF `cls` does not exactly match one of the Target_Class names, or `limit` is not an integer from 1 to 100 inclusive, THEN THE API_Server SHALL respond with HTTP 422 and a JSON error message identifying the invalid parameter, and SHALL not search the Vector_Index.
10. WHEN a search request completes with HTTP 200 or HTTP 422, THE API_Server SHALL append to the Query_Log exactly one line with the request time, query text, filters, result count (0 for rejected requests), and latency in milliseconds, without modifying existing lines.
11. WHEN a client requests `/`, THE API_Server SHALL serve the Console.
12. IF a valid search request matches no Hits, or the `camera` value matches no camera in the Metadata_Store, or the Allowed_ID_Set for the Search_Filter is empty, THEN THE API_Server SHALL respond with HTTP 200 and an empty JSON list.
13. IF a client sends `GET /api/search` while the Embedder model is not loaded or the Vector_Index cannot be read, THEN THE API_Server SHALL respond with HTTP 503 and a JSON error message indicating that search is unavailable, SHALL return no Events, and SHALL leave the Metadata_Store and Vector_Index unchanged.
14. IF appending a line to the Query_Log fails, THEN THE API_Server SHALL still return the search response it would otherwise have returned, and SHALL keep accepting later search requests.

### Requirement 11: Media Serving

**User Story:** As an Operator, I want videos to stream and seek in the browser, so that I can jump straight to the incident.

#### Acceptance Criteria

1. WHEN a client sends `GET /media/video/{video_id}` without a Range header, and `video_id` exists in the Metadata_Store and its Playback_File exists under `data/playback/`, THE API_Server SHALL respond with HTTP 200, the full Playback_File as the body, `Content-Type: video/mp4`, `Accept-Ranges: bytes`, and a `Content-Length` equal to the Playback_File size in bytes.
2. WHEN a client sends `GET /media/video/{video_id}` with a single-range header `Range: bytes=a-b`, where a and b are non-negative integers, a ≤ b, and a is less than the Playback_File size, THE API_Server SHALL respond with HTTP 206, `Content-Type: video/mp4`, a `Content-Range: bytes a-e/size` header where e is the smaller of b and (size − 1), a `Content-Length` of (e − a + 1), and exactly bytes a through e (zero-based, inclusive) of the Playback_File.
3. WHEN a client sends `GET /media/video/{video_id}` with an open-ended header `Range: bytes=a-`, where a is a non-negative integer less than the Playback_File size, THE API_Server SHALL treat the end as byte (size − 1) and respond as specified in criterion 2.
4. THE API_Server SHALL return 206 response bodies such that, for any Playback_File of 1 byte or more and any split of the byte range 0 to (size − 1) into consecutive, non-overlapping ranges requested in ascending order, joining the response bodies reproduces the Playback_File byte for byte (round-trip property).
5. IF a Range header of the form `bytes=a-b` or `bytes=a-` has a start a that is at or beyond the Playback_File size, or a header of the form `bytes=-n` has n equal to 0, THEN THE API_Server SHALL respond with HTTP 416, a `Content-Range: bytes */size` header, and a body that contains no bytes of the Playback_File.
6. IF `video_id` does not exist in the Metadata_Store, or it exists but its Playback_File is missing from `data/playback/`, THEN THE API_Server SHALL respond with HTTP 404 and a body that contains no video bytes.
7. WHEN a client sends `GET /media/thumbs/{name}` and `name` matches an existing Thumbnail file in `data/thumbs/`, THE API_Server SHALL respond with HTTP 200, the full Thumbnail file as the body, and `Content-Type: image/jpeg`.
8. IF a requested media path, after URL-decoding and path normalisation (including `..` segments, encoded separators, and absolute paths), resolves outside `data/thumbs/` or `data/playback/`, or the requested Thumbnail name does not match an existing file in `data/thumbs/`, THEN THE API_Server SHALL respond with HTTP 404 and a body that contains no file bytes.
9. WHEN a client sends `GET /media/video/{video_id}` with a suffix header `Range: bytes=-n`, where n is a positive integer, THE API_Server SHALL respond with HTTP 206 and the last n bytes of the Playback_File (or the whole file if n is at least the file size), with `Content-Range` and `Content-Length` headers that match the returned bytes as specified in criterion 2.
10. IF a Range header does not match the form `bytes=a-b`, `bytes=a-`, or `bytes=-n` with non-negative integers, or has a greater than b, or specifies more than one range, THEN THE API_Server SHALL ignore the Range header and respond as specified in criterion 1.

### Requirement 12: Operator Console

**User Story:** As an Operator, I want a simple search console, so that I can find and review incidents without training or extra software.

#### Acceptance Criteria

1. THE Console SHALL be a single page of plain HTML, JavaScript, and CSS served by the API_Server, with no build step and no externally hosted assets.
2. THE Console SHALL provide a query input that accepts at most 256 characters, a camera dropdown filled from `GET /api/cameras` with an "All cameras" option selected by default, start and end datetime inputs that are empty by default, and a class filter listing exactly the six Target_Classes plus an "All classes" option selected by default.
3. WHEN the Operator submits a query by activating the search control or pressing Enter in the query input, THE Console SHALL call `GET /api/search` with the query text trimmed of leading and trailing whitespace and only the filters that are set (camera other than "All cameras", non-empty start, non-empty end, class other than "All classes"), show a loading indicator, and disable the search control until the response arrives or the request fails.
4. WHEN search results arrive, THE Console SHALL replace any previous results with one card per returned Event, in the order returned by the API_Server, each showing the Thumbnail, camera label, absolute start and end time formatted as YYYY-MM-DD HH:MM:SS from the ISO 8601 Absolute_Timestamps returned by the API_Server, and the Event_Score rounded to 3 decimal places.
5. WHEN a search returns no Events, THE Console SHALL clear any previous results and show the message "No matching incidents found".
6. IF a search request returns an error response, fails due to a network error, or receives no response within 30 seconds, THEN THE Console SHALL hide the loading indicator, re-enable the search control, keep the entered query and filters unchanged, and show the error message returned by the API_Server or, when none is returned, an error message indicating that the server could not be reached or did not respond in time.
7. WHEN the Operator activates a result card, THE Console SHALL load the Event's Playback_File and, after the `loadedmetadata` event, set the player `currentTime` to the event start Frame_Offset returned by the API_Server, such that the read-back `currentTime` is within 0.5 seconds of that offset.
8. WHILE an Event is loaded in the player, THE Console SHALL highlight on the player timeline the span from the event start offset to the event end offset, positioned proportionally to the Playback_File duration, and mark exactly one result card, the active Event's card, as active.
9. WHILE an Event is loaded in the player, WHEN the Operator activates the replay control, THE Console SHALL set the player `currentTime` to within 0.5 seconds of the event start offset of the active Event.
10. WHEN the Operator activates the next or previous control, THE Console SHALL load the Event immediately after or before the active Event in the result list order, and THE Console SHALL keep the previous control disabled while the first Event is active, the next control disabled while the last Event is active, and the replay, next, and previous controls disabled while no Event is loaded.
11. THE Console SHALL make every control, including each result card, reachable with the Tab key in visual order and operable with Enter or Space, and SHALL show a focus outline at least 2 CSS pixels thick on the focused element.
12. THE Console SHALL give every form control a programmatic label, every Thumbnail an alt text naming the camera label and absolute start time, and SHALL expose the loading indicator, the no-results message, and every error message as status announcements to assistive technologies.
13. IF the query text is empty, consists only of whitespace, or exceeds 256 characters, THEN THE Console SHALL not send a search request, SHALL show a validation message next to the query input indicating the allowed length of 1 to 256 characters, and SHALL keep the entered query and filters unchanged.
14. IF the `GET /api/cameras` request fails or receives no response within 30 seconds, THEN THE Console SHALL offer only the "All cameras" option in the camera dropdown, show an error message indicating that the camera list could not be loaded, and keep search available.
15. IF the Playback_File of an activated Event fails to load, THEN THE Console SHALL show an error message in the player area indicating that the video could not be loaded, keep the result list and its order unchanged, and keep the failed Event's card marked as active.

### Requirement 13: Offline Operation and Model Provisioning

**User Story:** As an Integrator at an air-gapped facility, I want NAB_Sentry to run with no network access, so that footage and queries never leave the premises.

#### Acceptance Criteria

1. WHEN the Integrator runs the Model_Fetcher on a machine with internet access, THE Model_Fetcher SHALL download the OpenCLIP ViT-B/32 `laion2b_s34b_b79k` weights and the YOLO11n weights, export YOLO11n to ONNX, and store all resulting model files under `models/`.
2. WHEN the Model_Fetcher has downloaded every model file and the YOLO11n ONNX export has completed, THE Model_Fetcher SHALL write a manifest under `models/` that lists every model file NAB_Sentry loads at runtime, each with its path relative to `models/` and its SHA-256 hash as a 64-character hexadecimal string, and SHALL exit with status code 0.
3. WHEN NAB_Sentry starts, THE NAB_Sentry SHALL set `HF_HUB_OFFLINE=1` before importing any model library and load every model only from local paths under `models/`.
4. WHEN NAB_Sentry starts, THE NAB_Sentry SHALL check every file listed in the manifest against its manifest SHA-256 hash before the API_Server accepts its first request and before the Ingest_Pipeline processes its first frame, and SHALL finish this check within 30 seconds on the Target_Hardware.
5. IF the manifest is missing, a model file listed in the manifest is missing, or a model file's SHA-256 hash does not match the manifest, THEN THE NAB_Sentry SHALL exit with a non-zero status code without accepting API_Server requests or starting ingest, and SHALL show an error message naming each failing file and the command to run the Model_Fetcher.
6. WHILE NAB_Sentry is running startup, ingest, search, or media serving, THE NAB_Sentry SHALL open zero network connections to any address other than the loopback interface.
7. THE NAB_Sentry SHALL write all footage, Thumbnails, Playback_Files, Metadata_Store files, Vector_Index files, the Query_Log, and other log files only inside the local workspace directories `data/` and `models/`.
8. IF any model download fails after 3 attempts or the YOLO11n ONNX export fails, THEN THE Model_Fetcher SHALL exit with a non-zero status code, show an error message naming the file that failed, and leave no manifest under `models/`, so a partial download cannot pass the startup check.
9. WHILE the host has no active network interface other than loopback, THE NAB_Sentry SHALL complete startup, ingest a video file, return search results, and serve Thumbnails and Playback_Files without network-related errors.
10. WHEN the Console loads in the Operator's browser, THE Console SHALL request scripts, stylesheets, fonts, images, and media only from the API_Server origin.

### Requirement 14: Performance on Target Hardware

**User Story:** As an Integrator, I want known throughput and latency on a CPU-only laptop, so that I can size deployments and run a smooth demo.

#### Acceptance Criteria

1. WHEN the Integrator runs the Benchmark_Tool with an input video file, THE Benchmark_Tool SHALL run 5 untimed warm-up iterations and then at least 50 timed iterations per stage, and report the median and 95th-percentile values of: Embedder milliseconds per image at batch size 1, Embedder milliseconds per image at batch size 8, Detector milliseconds per frame, and Motion_Gate frames per second.
2. WHEN the Benchmark_Tool finishes its timed iterations, THE Benchmark_Tool SHALL report the projected ingest minutes per hour of footage at the configured Sample_Rate, computed from the measured median per-stage times and the fraction of sampled frames that passed the Motion_Gate in the input video, and SHALL state the Sample_Rate and pass fraction used.
3. WHILE running on the Target_Hardware with a Vector_Index of up to 200,000 vectors and Top_K at its configured value, THE Search_Engine SHALL return search results within 1.5 seconds at the 95th percentile over at least 100 sequential search requests from a single client, with and without a Search_Filter, excluding the first request after startup, measured at the API_Server from request receipt to response sent.
4. WHILE ingesting on the Target_Hardware at a Sample_Rate of 1 with at most 50% of sampled frames passing the Motion_Gate, THE Ingest_Pipeline SHALL complete one hour of 1920x1080 footage at up to 30 frames per second within 60 minutes of wall-clock time, measured from ingest start until the Metadata_Store rows, Vector_Index vectors, Thumbnails, and Playback_File for that video are all stored.
5. WHILE ingesting or searching on the Target_Hardware with a Vector_Index of up to 200,000 vectors, THE NAB_Sentry SHALL keep peak resident memory of all NAB_Sentry processes combined below 6 GB.
6. WHEN the API_Server starts with a Vector_Index of up to 200,000 vectors on the Target_Hardware, THE API_Server SHALL load the Detector model, the Embedder model, and the Vector_Index and return a successful response to its first search request within 30 seconds of process launch.
7. IF a model file in `models/` or the Vector_Index is missing or cannot be loaded when the API_Server starts, THEN THE API_Server SHALL stop without accepting requests, display an error message identifying the file that failed to load, and exit with a non-zero exit code.
8. IF the Benchmark_Tool cannot load a model or read the input video file, THEN THE Benchmark_Tool SHALL display an error message identifying the stage and file that failed, skip timing for that stage, and exit with a non-zero exit code after reporting the stages that did complete.

### Requirement 15: Synthetic Test Data

**User Story:** As a developer, I want generated test videos with exact ground truth, so that pipeline correctness can be checked automatically without real footage.

#### Acceptance Criteria

1. WHEN the Synthetic_Generator runs with default settings, THE Synthetic_Generator SHALL write one 90-second (900-frame), 10 fps, 640×480 pixel MP4 video whose background is a seed-derived texture that is pixel-identical in every frame, with a solid red square (side at least 60 pixels) visible and changing position in every frame with Frame_Offset from 10 s (inclusive) to 25 s (exclusive), a solid blue circle (diameter at least 60 pixels) visible and changing position in every frame with Frame_Offset from 50 s (inclusive) to 65 s (exclusive), and no object visible in any other frame.
2. WHEN the Synthetic_Generator writes a video, THE Synthetic_Generator SHALL write a `ground_truth.json` in the same output directory that lists, for each object, its label (`red square` or `blue circle` for the default video), its start offset and end offset in seconds as Frame_Offsets, and the camera ID and Video_Start_Time written to that video's Sidecar_File.
3. WHEN the Synthetic_Generator writes a video, THE Synthetic_Generator SHALL write a Sidecar_File for that video containing `camera_id`, `label`, and `start_time` (ISO 8601), and SHALL name the video file according to the Filename_Convention using the same camera ID and Video_Start_Time, so that both metadata sources resolve to identical values.
4. FOR ALL generated videos, running the Synthetic_Generator twice with the same seed and settings SHALL produce videos with identical frame count and identical decoded pixel values for every frame, and an identical `ground_truth.json` (determinism property).
5. WHEN the default synthetic video is ingested with default Config values, THE Motion_Gate SHALL pass at least one sampled frame whose Frame_Offset lies within each ground-truth object interval.
6. WHEN the default synthetic video is ingested with default Config values, THE Motion_Gate SHALL pass, outside the ground-truth object intervals, only the first sampled frame, Keyframe_Interval keyframes, and at most one sampled frame within 1 second after each interval end (the frame showing the object's disappearance).
7. WHEN the default synthetic video is searched with the query "red square" and, separately, with the query "blue circle", THE Search_Engine SHALL return at least one Event, and the top-ranked Event SHALL have a time window (start to end Frame_Offset, including Event_Padding) that overlaps the ground-truth interval of the object named in the query by at least 1 second.
8. IF the Synthetic_Generator is given an invalid setting (duration or fps of zero or less, or an object interval that starts before 0 s, ends after the video duration, or ends at or before its start), THEN THE Synthetic_Generator SHALL exit with a non-zero status, print an error message identifying the invalid setting, and write no video, Sidecar_File, or `ground_truth.json`.
9. IF the Synthetic_Generator cannot write to the output directory, THEN THE Synthetic_Generator SHALL exit with a non-zero status, print an error message indicating the output directory is not writable, and leave no partially written video, Sidecar_File, or `ground_truth.json`.

### Requirement 16: Retrieval Quality Evaluation

**User Story:** As an Integrator, I want repeatable quality metrics on real footage, so that I can tune parameters and show measured evidence to customers.

#### Acceptance Criteria

1. THE Evaluator SHALL read `eval/queries.yaml`, which holds between 15 and 20 queries inclusive, each with a query text of 1 to 200 characters, a ground-truth camera ID, a ground-truth start time, and a ground-truth end time, where both times are ISO 8601 Absolute_Timestamps and the end time is later than the start time.
2. WHEN the Evaluator runs a query, THE Evaluator SHALL submit the query text to the Search_Engine with no Search_Filter and with the Top_K value from the Config, and SHALL count a returned Event as relevant if and only if the Event's camera ID equals the ground-truth camera ID and the Event's padded Absolute_Timestamp window (start including Event_Padding through end including Event_Padding) satisfies Event start ≤ ground-truth end time AND Event end ≥ ground-truth start time.
3. WHEN the Evaluator completes a run, THE Evaluator SHALL report for each query the query text, precision@5 (the number of relevant Events among the first 5 ranked Events divided by 5, where missing ranks count as not relevant), hit@1 (1 if the rank-1 Event is relevant, otherwise 0, and 0 if no Events are returned), and search latency in milliseconds (wall-clock time from query submission to receipt of the ranked Events), and SHALL report the arithmetic mean of precision@5, hit@1, and search latency across all queries not reported as errors, together with the number of queries run and the number of queries reported as errors.
4. THE Evaluator SHALL compute precision@5 and hit@1 so that, for every query result list including an empty list, each value lies between 0 and 1 inclusive, precision@5 is a multiple of 0.2, and hit@1 = 1 implies precision@5 ≥ 0.2 (metamorphic property).
5. THE Config SHALL hold every tunable parameter: Sample_Rate, Motion_Threshold, Keyframe_Interval, gate width, detection threshold, per-frame detection maximum, batch size, Top_K, Merge_Gap, Event_Padding, and Label_Boost, with each parameter defined exactly once in the Config and read from the Config by the Ingest_Pipeline, Search_Engine, and Evaluator.
6. WHEN the Evaluator completes a run, THE Evaluator SHALL write a single report containing the run date and time, the name and value of every Config parameter listed in criterion 5 as used for that run, the aggregate metrics, and the per-query results.
7. IF a query in `eval/queries.yaml` references a camera ID that is not in the Metadata_Store, lacks query text, lacks a ground-truth field, or has a ground-truth end time that is not later than its start time, THEN THE Evaluator SHALL record that query in the report as an error stating the reason, exclude it from the aggregate metrics, and continue with the remaining queries.
8. IF `eval/queries.yaml` is missing, cannot be parsed, holds fewer than 15 or more than 20 queries, or every query in it is reported as an error, THEN THE Evaluator SHALL output an error message indicating the cause, report no aggregate metrics, and exit with a non-zero exit status.
9. WHEN the Evaluator is run twice against the same Metadata_Store, Vector_Index, `eval/queries.yaml`, and Config values, THE Evaluator SHALL report identical per-query precision@5 and hit@1 values and identical aggregate precision@5 and hit@1 values in both runs.

### Requirement 17: Demo Readiness

**User Story:** As a NAB AI presenter, I want a one-command demo on pre-indexed footage, so that I can show NAB_Sentry reliably to prospects with no network.

#### Acceptance Criteria

1. WHEN the Integrator runs `run_demo.ps1` from the workspace root and both the Metadata_Store and the Vector_Index exist, THE NAB_Sentry SHALL activate the existing `venv/`, start the API_Server bound only to 127.0.0.1 without running the Ingest_Pipeline, and, within 30 seconds of the script starting on the Target_Hardware, print the Console URL including the port once `GET /api/health` reports that the models are loaded.
2. IF the Metadata_Store or the Vector_Index is absent when `run_demo.ps1` runs, THEN THE NAB_Sentry SHALL run the Ingest_Pipeline on every video file in `data/videos/`, print the number of videos ingested and the number that failed, and only after ingest finishes start the API_Server and print the Console URL as in criterion 1.
3. WHILE all host network adapters are disabled, WHEN the Integrator submits any query from `eval/queries.yaml` in the Console, THE NAB_Sentry SHALL display a ranked list of Events, each showing the camera ID, ISO 8601 absolute start and end timestamps, and a Thumbnail, with no network-related error shown in the Console.
4. THE NAB_Sentry SHALL include a README at the workspace root with a separate section for each of: model provisioning with the Model_Fetcher, ingest, running the demo with `run_demo.ps1`, running the Evaluator, and the MVP security limitations; each of the first four sections SHALL give the exact PowerShell command to run, and the security section SHALL state that there is no authentication, no audit logging, and that the API_Server binds only to 127.0.0.1.
5. WHILE all host network adapters are disabled, WHEN the Integrator selects an Event in the Console, THE Console SHALL play the Event's Playback_File in the HTML5 player starting within 1 second of the Event's start Frame_Offset.
6. IF `venv/` does not exist, a model file is missing or fails the manifest hash check, or the API_Server cannot bind to its port on 127.0.0.1, THEN THE NAB_Sentry SHALL print an error message naming the failed check and the corrective step, exit `run_demo.ps1` with a non-zero exit code, and not print the Console URL.
7. IF ingest under criterion 2 finds no video files in `data/videos/` or finishes with zero vectors in the Vector_Index, THEN THE NAB_Sentry SHALL print an error message naming `data/videos/`, exit `run_demo.ps1` with a non-zero exit code, and not start the API_Server.

### Requirement 18: MVP Security Posture

**User Story:** As a facility security officer, I want the MVP's security limits stated and contained, so that I can judge whether it is safe to pilot.

#### Acceptance Criteria

1. THE API_Server SHALL listen only on the IPv4 loopback address 127.0.0.1 and on no other IPv4 or IPv6 address, so that a connection attempt to the API_Server port from another host, or to a non-loopback address of the Target_Hardware, is refused.
2. WHEN the API_Server starts, THE API_Server SHALL write a warning-level message to its console log before it accepts the first request. The message SHALL state that authentication, access control, and audit logging are not enabled in the MVP and that the Query_Log is not tamper-evident.
3. THE README SHALL contain a security limitations section stating that (a) the MVP has no authentication or access control, (b) the API_Server listens only on 127.0.0.1, (c) the Query_Log is a local file that is not tamper-evident, and (d) access control and a tamper-evident audit log are planned after the MVP.
4. THE API_Server SHALL pass every request value used in a Metadata_Store query (including camera ID, start and end timestamps, class, and limit) only as a bound parameter of a parameterised SQL statement. A request value that contains SQL syntax, such as a single quote, a semicolon, or a comment marker, SHALL be treated as a literal value and SHALL leave all Metadata_Store tables and rows unchanged.
5. IF the API_Server is configured or launched with a bind address other than 127.0.0.1 (for example 0.0.0.0 or a LAN address), THEN THE API_Server SHALL exit before it starts listening, with an error stating that the MVP permits only loopback binding.
6. IF a media request names a file that does not resolve to a location inside `data/thumbs/` or `data/playback/` (including names that contain `..` segments, absolute paths, or drive letters), THEN THE API_Server SHALL reject the request with an error response and SHALL return no file content.
7. IF an unhandled error occurs while the API_Server processes a request, THEN THE API_Server SHALL respond with an error message that contains no stack trace, no absolute file system path, and no SQL text, and SHALL continue to serve later requests.
