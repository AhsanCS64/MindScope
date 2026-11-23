# Deploy MindScope to Render and Vercel

The source includes a Render Blueprint, a Vercel Vite configuration, pinned model artifacts, and a version-pinned data preparation command.

## GitHub

Create or use a repository you control and put the project root at its root. Commit the portable booster and preprocessing files, model metadata, global SHAP figures, lockfile, and source. The ignore rules exclude the full survey dataset, original joblib, local environments, and generated builds. The source ZIP is for transport. Extract it before pushing.

## Render

1. Create a Blueprint from the GitHub repository containing `render.yaml`. The Blueprint selects the free web plan and Python 3.13.5.
2. The Blueprint sets `CORS_ORIGINS` to `https://mind-scope-tau.vercel.app`. If you use a different frontend domain, update the Blueprint to that exact HTTPS origin without a trailing slash. A manually created Render web service needs the same value set in its Environment tab.
3. Use the included health path `/api/health`. The build installs the pinned dependencies and downloads the verified OpenML ARFF to build the local CSV. At runtime there is one CPU worker and no database.
4. Confirm a successful build and open `https://<render-service>.onrender.com/api/health`.

Render's free web service tier is listed at 512 MB and spins down after 15 minutes without traffic. The platform documents a wake-up time of about a minute. Check the current service metrics after an idle wake-up and explanation request. [Render free services](https://render.com/docs/free) · [Blueprint reference](https://render.com/docs/blueprint-spec)

## Vercel

1. Import the same GitHub repository and set the root directory to `frontend`.
2. Select Vite, set the build command to `npm run build -- --mode vercel`, and use `dist` as the output directory. The committed `frontend/vercel.json` sets both values.
3. Deploy. The Vercel build uses same-origin `/api` requests. Vercel forwards them to the Render origin in `frontend/vercel.json`. Update the rewrite destination if the Render service URL changes.
4. Open `/api/health` on your Vercel domain. A successful response confirms the frontend host can reach Render through the rewrite. [Vercel rewrites](https://vercel.com/docs/routing/rewrites)

## Public smoke checks

- Verify `/api/health`, `/api/model/info`, and the OpenAPI documentation on Render.
- Open the Vercel frontend and confirm that the Overview returns 27,901 source records.
- Open Assessment, load the hypothetical profile, run the model, and confirm that the probabilities sum to one and a SHAP waterfall appears.
- Verify the dataset correlations and SHAP PNGs. Submit an incomplete request and confirm a 422 response with no submitted values echoed.
- After the backend has been idle, repeat the smoke test and record cold-start time and peak RAM from Render.

