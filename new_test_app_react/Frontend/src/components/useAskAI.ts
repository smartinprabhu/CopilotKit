import { useState } from "react";
import axios from "axios";
import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';

  const WEBAPPAPIURL =  `${AppConfig.API_URL}/`;

export function useAskAI(parametersList: any[] = []) {
  const [aiModal, setAIModal] = useState(false);
  const [aiLoading, setAILoading] = useState(false);
  const [aiData, setAIData] = useState<string | JSX.Element>("");

  const onOpenAI = async (obj: any) => {
    setAIModal(true);
    setAILoading(true);
    setAIData("");

    if (obj?.cards && obj?.cards?.stats) {
      const enrichedStats = obj.cards.stats.map((stat: any) => {
        const oldKpi = obj.cards.oldStats.find((old: any) => old.key === stat.key);
        const parameterData = parametersList.find(p => p.name === stat.key);
        const kpiDirection = parameterData?.direction ?? "Low is Good";

        const oldValue = oldKpi?.total ?? 0;
        const currentValue = stat.total;

        let change = 0;
        let changeText = "";
        let invertChange = false;

        if (oldValue > 0) {
          change = ((currentValue - oldValue) / oldValue) * 100;
          changeText = `${change > 0 ? "increase" : "decrease"}`;
          invertChange = change > 0;
        } else if (currentValue > 0) {
          change = 100;
          changeText = "↑ New";
          invertChange = true;
        }

        return {
          ...stat,
          changeValue: {
            percent: Number(change.toFixed(2)),
            text: changeText,
            direction: kpiDirection,
          },
        };
      });

      delete obj.cards.oldStats;
      obj.cards.stats = enrichedStats;
    }

    try {
      const params = new URLSearchParams({
        authtoken: AuthService.getAccessToken(),
        domain: "https://support-api-dev.zentere.com",
        json_data: JSON.stringify(obj),
      });

      const response = await axios.get(
        `${WEBAPPAPIURL}webhook/c2d5f5a2-ce78-480d-86d8-832d64b93d9d?${params.toString()}`,
        {
          headers: {
            Authorization: `Bearer ${AuthService.getAccessToken()}`,
          },
        }
      );

      setAIData(response?.data?.[0]?.output || []);
    } catch (error) {
      console.error("Error fetching data:", error);
      setAIData("");
    } finally {
      setAILoading(false);
    }
  };

  return {
    aiModal,
    setAIModal,
    aiLoading,
    aiData,
    onOpenAI,
  };
}
